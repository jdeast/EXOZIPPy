// genulens_reference_driver.cpp -- evaluate genulens's OWN model functions at
// chosen points and print them as JSON, for EXOZIPPy's point-by-point
// validation of its port of the genulens Galactic model.
//
// genulens: Koshimoto, Baba & Bennett 2021, ApJ 917, 78 ("K21"),
// https://github.com/nkoshimoto/genulens (GPLv3).  This file contains no
// genulens code: it calls genulens's functions through its public headers
// and links against the genulens_core library a user builds from genulens's
// own source.  Nothing of genulens is vendored into EXOZIPPy; the recorded
// numbers (tests/fixtures/genulens_points_*.json) are its outputs.
//
// Build (from a genulens checkout built with CMake, see
// scripts/make_genulens_points.py, which compiles and runs this):
//
//   g++ -std=c++17 -O2 -I<genulens>/src genulens_reference_driver.cpp \
//       <genulens>/build/libgenulens_core.a -lgsl -lgslcblas -lm \
//       [-lstdc++fs] -o genulens_reference_driver
//
// Run from the genulens checkout root (it reads input_files/ relative to the
// working directory, as genulens itself does):
//
//   genulens_reference_driver l <deg> b <deg> DLIST <d1,d2,...> \
//       [NMC <n>] [seed <s>] [AIrc <mag> Isrange <lo> <hi>] [NSD <0|1>]
//
// Every genulens model option (DISK, model, addX, R0, ...) passes through to
// genulens's own option reader unchanged, so the driver evaluates the model
// genulens would simulate with the same command line.  The initialization
// sequence mirrors simulation/sampler.cpp (genulens 2.0.0a4): model options,
// spatial frame, NSD switch (|l| < 5 and |b| < 2 turn the NSD on), mass
// function and population normalizations, luminosity functions, Shu tables,
// density normalization.

#include "genulens/cli/option.h"
#include "genulens/io/input_data.hpp"
#include "genulens/simulation/initialize.hpp"
#include "genulens/simulation/internal/runtime.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define fopen(path, mode) genulens::open_input_file((path), (mode))

namespace genulens {
// Definitions live in libgenulens_core (simulation/*.cpp); runtime.hpp
// declares some of them with older signatures, so declare the ones used here.
void calc_rho_each(RunContext &ctx, double D, int idata, double *rhos, double *xyz, double *xyb);
void calc_sigvb(RunContext &ctx, double xb, double yb, double zb, double *sigvbs);
void get_vxyz_ran(RunContext &ctx, double *vxyz, int i, double tau, double D, double lD, double bD);
double calc_PRRg(RunContext &ctx, int R, int z, double fg, double sigU0, double hsigU, int rd);
void Mini2Mrem(RunContext &ctx, double *pout, double Mini, int mean);
double fLF_detect(RunContext &ctx, double extI, double Imin, double Imax, int idisk);
}  // namespace genulens

using namespace genulens;

namespace {

std::string string_option(int argc, char **argv, const char *name)
{
    for (int i = 1; i + 1 < argc; ++i)
        if (std::strcmp(argv[i], name) == 0) return argv[i + 1];
    return "";
}

std::vector<double> parse_list(const std::string &s)
{
    std::vector<double> out;
    std::size_t start = 0;
    while (start < s.size()) {
        std::size_t end = s.find(',', start);
        if (end == std::string::npos) end = s.size();
        out.push_back(std::atof(s.substr(start, end - start).c_str()));
        start = end + 1;
    }
    return out;
}

// --- tiny JSON writer (numbers at full precision) -------------------------
bool first_item = true;
void sep()
{
    if (!first_item) std::printf(",");
    first_item = false;
}
void key(const char *k)
{
    sep();
    std::printf("\"%s\":", k);
    first_item = true;
}
void open_obj() { std::printf("{"); first_item = true; }
void close_obj() { std::printf("}"); first_item = false; }
void open_arr() { std::printf("["); first_item = true; }
void close_arr() { std::printf("]"); first_item = false; }
void num(double v)
{
    sep();
    if (std::isfinite(v)) std::printf("%.17g", v);
    else std::printf("null");
}
void kv(const char *k, double v) { key(k); num(v); }
template <typename T>
void karr(const char *k, const T *v, int n)
{
    key(k);
    open_arr();
    for (int i = 0; i < n; ++i) num(static_cast<double>(v[i]));
    close_arr();
}
template <typename T>
void karr(const char *k, const T &v)
{
    key(k);
    open_arr();
    for (const auto &x : v) num(static_cast<double>(x));
    close_arr();
}

// Polar velocity in genulens's frame (x from the GC toward the Sun, y toward
// l = 90 deg): get_vxyz_ran builds vx = -vphi y/R + vR x/R,
// vy = vphi x/R + vR y/R, so vR = (x vx + y vy)/R, vphi = (x vy - y vx)/R.
void polar(double x, double y, const double *v, double *vr, double *vphi)
{
    const double R = std::sqrt(x * x + y * y);
    *vr = (x * v[0] + y * v[1]) / R;
    *vphi = (x * v[1] - y * v[0]) / R;
}

std::vector<double> quantiles(std::vector<double> v, const std::vector<double> &p)
{
    std::sort(v.begin(), v.end());
    std::vector<double> q;
    for (double pi : p) {
        const double pos = pi * (v.size() - 1);
        const std::size_t lo = static_cast<std::size_t>(std::floor(pos));
        const std::size_t hi = std::min(lo + 1, v.size() - 1);
        q.push_back(v[lo] + (pos - lo) * (v[hi] - v[lo]));
    }
    return q;
}

}  // namespace

int main(int argc, char **argv)
{
    // ---- model, exactly as Sampler::prepare does it -------------------------
    RunContext ctx = Initializer().create_context();
    Initializer().initialize_rng(ctx, argc, argv);
    Initializer().read_model_options(ctx, argc, argv);
    Initializer().finalize_spatial_model(ctx, argc, argv);

    const double l = getOptiond(argc, argv, "l", 1, 1.0);
    const double b = getOptiond(argc, argv, "b", 1, -3.9);
    const double l_signed = model::wrapped_longitude_deg(l);
    ctx.density.lDs = (double *)std::malloc(sizeof(double));
    ctx.density.bDs = (double *)std::malloc(sizeof(double));
    ctx.density.lDs[0] = l;
    ctx.density.bDs[0] = b;

    const double Isst = getOptiond(argc, argv, "Isrange", 1, 14.0);
    const double Isen = getOptiond(argc, argv, "Isrange", 2, 21.0);
    const double AIrc = getOptiond(argc, argv, "AIrc", 1, 0.0);
    double DMrc = getOptiond(argc, argv, "DMrc", 1, 0.0);
    const double hdust = getOptiond(argc, argv, "hdust", 1, 164.0);

    PopulationRuntime pop;
    pop.initialize_mass_function(ctx, ctx.imf_options);
    pop.initialize_luminosity_functions(ctx, Isst, Isen, 0.0, 0.0, AIrc, 0.0);

    KinematicRuntimeTables shu;
    shu.initialize_shu_distribution(ctx);

    for (int i = 0; i < 2; ++i)
        ctx.density.y0d.data()[i] = std::exp(-ctx.density.R0 / ctx.density.Rd.data()[i]);

    if (std::fabs(l_signed) < 5 && std::fabs(b) < 2) ctx.density.ND = 3;
    ctx.density.ND = getOptiond(argc, argv, "NSD", 1, ctx.density.ND);
    if (ctx.density.ND > 0) ctx.density.ND = 3;
    if (ctx.density.ND) ctx.density.rho0ND = 1;
    Initializer().finalize_density_normalization(ctx);
    NsdMomentRuntime nsd;
    nsd.initialize_if_enabled(ctx);

    // Extinction as Sampler builds it (ExponentialDustExtinction with the
    // legacy AIrc reference): A_I(D) = A_I,rc (1 - e^{-D/h}) / (1 - e^{-Dmean/h}).
    if (DMrc == 0) DMrc = 14.3955 - 0.0239 * l_signed + 0.0122 * std::fabs(b) + 0.128;
    const double sinb = std::sin(b / 180.0 * PI);
    const double hscale = hdust / (std::fabs(sinb) + 0.0001);
    const double Dmean = std::pow(10, 0.2 * DMrc) * 10;
    const double ai0 = AIrc / (1.0 - std::exp(-Dmean / hscale));

    const std::vector<double> dlist = parse_list(string_option(argc, argv, "DLIST"));
    const int nmc = getOptiond(argc, argv, "NMC", 1, 20000);
    const std::vector<double> probs = {0.02, 0.05, 0.1, 0.16, 0.25, 0.5, 0.75, 0.84, 0.9, 0.95, 0.98};

    open_obj();
    key("sightline");
    open_obj();
    kv("l", l); kv("b", b); kv("ND", ctx.density.ND);
    kv("AIrc", AIrc); kv("Isst", Isst); kv("Isen", Isen); kv("DMrc", DMrc);
    kv("hscale", hscale); kv("Dmean", Dmean); kv("AI0", ai0);
    close_obj();

    // ---- constants: every number the port must reproduce --------------------
    const auto &d = ctx.density;
    const auto &k = ctx.kinematics;
    key("constants");
    open_obj();
    kv("R0", d.R0); kv("thetaD", d.thetaD);
    karr("xyzSgrA", ctx.xyzSgrA);
    kv("vxsun", k.vxsun); kv("vysun", k.vysun); kv("vzsun", k.vzsun);
    karr("rho0d", d.rho0d); karr("n0d", d.n0d); karr("n0MSd", d.n0MSd); karr("n0RGd", d.n0RGd);
    karr("zd", d.zd); karr("Rd", d.Rd); karr("y0d", d.y0d); kv("Rdbreak", d.Rdbreak);
    kv("rho0b", d.rho0b); kv("n0MSb", d.n0MSb); kv("n0b", d.n0b); kv("n0RGb", d.n0RGb);
    kv("fb_MS", d.fb_MS); kv("m2nb_MS", d.m2nb_MS); kv("m2nb_WD", d.m2nb_WD);
    kv("rho0ND", d.rho0ND); kv("n0MSND", d.n0MSND); kv("n0ND", d.n0ND);
    kv("rho0SH", d.rho0SH); kv("n0MSSH", d.n0MSSH); kv("n0SH", d.n0SH);
    karr("medtauds", k.medtauds);
    karr("Rcs", k.Rcs.data(), k.nVcs); karr("Vcs", k.Vcs.data(), k.nVcs);
    close_obj();

    // ---- density along the sight line ---------------------------------------
    key("density");
    open_arr();
    for (double D : dlist) {
        double rhos[11] = {}, xyz[3] = {}, xyb[2] = {};
        calc_rho_each(ctx, D, 0, rhos, xyz, xyb);
        double nMS[11], nall[11];
        for (int i = 0; i < 11; ++i) {
            nMS[i] = (i == 8) ? d.n0MSb * rhos[8] : (i == 9) ? d.n0MSND * rhos[9]
                   : (i == 10) ? d.n0MSSH * rhos[10] : d.n0MSd.data()[i] * rhos[i];
            nall[i] = (i == 8) ? d.n0b * rhos[8] : (i == 9) ? d.n0ND * rhos[9]
                    : (i == 10) ? d.n0SH * rhos[10] : d.n0d.data()[i] * rhos[i];
        }
        sep();
        open_obj();
        kv("D", D);
        karr("xyz", xyz, 3); karr("xyb", xyb, 2);
        karr("rhos", rhos, 11); karr("nMS", nMS, 11); karr("n", nall, 11);
        if (pop.has_luminosity_function) {
            // the grid's own extinction-plus-distance-modulus (los_density_grid.cpp)
            const double extI = ai0 * (1.0 - std::exp(-D / hscale)) + 5.0 * std::log10(0.1 * (D + 0.1));
            double fLF[11];
            for (int i = 0; i < 11; ++i) fLF[i] = fLF_detect(ctx, extI, Isst, Isen, i);
            kv("extI", extI);
            karr("fLF", fLF, 11);
        }
        close_obj();
    }
    close_arr();

    // ---- velocity: genulens's building blocks plus its own sampler -----------
    key("velocity");
    open_arr();
    for (double D : dlist) {
        double rhos[11] = {}, xyz[3] = {}, xyb[2] = {};
        calc_rho_each(ctx, D, 0, rhos, xyz, xyb);
        const double x = xyz[0], y = xyz[1], z = xyz[2];
        const double R = std::sqrt(x * x + y * y);
        for (int i = 0; i < 11; ++i) {
            if (i == 9 && !ctx.density.ND) continue;
            if (i == 9 && (R > ctx.nsd_moments.RenND || std::fabs(z) > ctx.nsd_moments.zenND)) continue;
            sep();
            open_obj();
            kv("D", D); kv("comp", i); kv("R", R); kv("z", z);
            if (i < 8) {
                const double tau = k.medtauds.data()[i];
                const double sigW0 = (i < 7) ? k.sigW10d * std::pow((tau + 0.01) / 10.01, k.betaW) : k.sigW0td;
                const double sigU0 = (i < 7) ? k.sigU10d * std::pow((tau + 0.01) / 10.01, k.betaU) : k.sigU0td;
                const double hsigW = (i < 7) ? k.hsigWt : k.hsigWT;
                const double hsigU = (i < 7) ? k.hsigUt : k.hsigUT;
                kv("sigU0", sigU0); kv("sigW0", sigW0); kv("hsigU", hsigU); kv("hsigW", hsigW);
                kv("sigU", sigU0 * std::exp(-(R - d.R0) / hsigU));
                kv("sigW", sigW0 * std::exp(-(R - d.R0) / hsigW));
                const int iz = (std::fabs(z) - k.zstShu) / k.dzShu;
                const int iR = (R > k.RstShu) ? (R - k.RstShu) / k.dRShu : 0;
                kv("iz", iz); kv("iR", iR);
                kv("z_node", k.zstShu + iz * k.dzShu); kv("R_node", k.RstShu + iR * k.dRShu);
            } else if (i == 8) {
                double sigvbs[3] = {};
                calc_sigvb(ctx, xyb[0], xyb[1], z, sigvbs);
                const double ct = d.costheta, st = d.sintheta;
                const double sigx = std::sqrt(sigvbs[0] * sigvbs[0] * ct * ct + sigvbs[1] * sigvbs[1] * st * st);
                const double sigy = std::sqrt(sigvbs[0] * sigvbs[0] * st * st + sigvbs[1] * sigvbs[1] * ct * ct);
                const double vrot = 0.001 * k.Omega_p * R;
                double avevxb = (xyb[1] > 0) ? -k.vx_str : k.vx_str;
                if (k.y0_str > 0) {
                    const double yn = std::fabs(xyb[1] / k.y0_str);
                    avevxb *= (1 - std::exp(-yn * yn));
                }
                const double mean[3] = {-vrot * y / R + avevxb * ct, vrot * x / R + avevxb * st, 0.0};
                karr("sigvb", sigvbs, 3);
                const double sig[3] = {sigx, sigy, sigvbs[2]};
                karr("sigma_xyz", sig, 3);
                karr("mean_xyz", mean, 3);
            } else if (i == 10) {
                const double sig[3] = {k.sigU_SH, k.sigV_SH, k.sigW_SH};
                karr("sigma_R_phi_z", sig, 3);
            }
            // genulens's own sampler (escape-speed rejection included)
            std::vector<double> vx(nmc), vy(nmc), vz(nmc), vr(nmc), vp(nmc);
            double m[3] = {}, c[3][3] = {};
            for (int j = 0; j < nmc; ++j) {
                double v[3] = {};
                const double tau = (i < 8) ? k.medtauds.data()[i] : 9.0;
                get_vxyz_ran(ctx, v, i, tau, D, l, b);
                vx[j] = v[0]; vy[j] = v[1]; vz[j] = v[2];
                polar(x, y, v, &vr[j], &vp[j]);
                for (int a = 0; a < 3; ++a) m[a] += v[a];
            }
            for (int a = 0; a < 3; ++a) m[a] /= nmc;
            for (int j = 0; j < nmc; ++j) {
                const double v[3] = {vx[j], vy[j], vz[j]};
                for (int a = 0; a < 3; ++a)
                    for (int bb = 0; bb < 3; ++bb) c[a][bb] += (v[a] - m[a]) * (v[bb] - m[bb]);
            }
            kv("nmc", nmc);
            karr("mc_mean_xyz", m, 3);
            key("mc_cov_xyz");
            open_arr();
            for (int a = 0; a < 3; ++a) {
                double row[3];
                for (int bb = 0; bb < 3; ++bb) row[bb] = c[a][bb] / (nmc - 1);
                sep(); open_arr(); for (double r : row) num(r); close_arr();
            }
            close_arr();
            karr("probs", probs);
            karr("mc_q_vR", quantiles(vr, probs));
            karr("mc_q_vphi", quantiles(vp, probs));
            karr("mc_q_vz", quantiles(vz, probs));
            close_obj();
        }
    }
    close_arr();

    // ---- the Shu DF: the unnormalized P(fg | R, z) and genulens's tables ----
    // Every node the velocity block touched, plus calc_PRRg on a fixed fg grid
    // at those nodes (the analytic formula, Sharma+14 / Schonrich+12).
    key("shu");
    open_arr();
    for (double D : dlist) {
        double rhos[11] = {}, xyz[3] = {}, xyb[2] = {};
        calc_rho_each(ctx, D, 0, rhos, xyz, xyb);
        const double R = std::sqrt(xyz[0] * xyz[0] + xyz[1] * xyz[1]);
        const int iz = (std::fabs(xyz[2]) - k.zstShu) / k.dzShu;
        const int iR = (R > k.RstShu) ? (R - k.RstShu) / k.dRShu : 0;
        const int Rn = k.RstShu + iR * k.dRShu, zn = k.zstShu + iz * k.dzShu;
        for (int i = 0; i < 8; ++i) {
            const double tau = k.medtauds.data()[i];
            const double hsigU = (i < 7) ? k.hsigUt : k.hsigUT;
            const int rd = (i == 0) ? d.Rd.data()[0] : (i < 7) ? d.Rd.data()[1] : d.Rd.data()[2];
            const double sigU0 = (i < 7) ? k.sigU10d * std::pow((tau + 0.01) / 10.01, k.betaU) : k.sigU0td;
            const int n = k.n_fgsShu[iz][iR][i];
            const double *fgs = k.fgsShu[iz][iR][i];
            const double *pn = k.PRRgShus[iz][iR][i];
            sep();
            open_obj();
            kv("comp", i); kv("R_node", Rn); kv("z_node", zn);
            kv("sigU0", sigU0); kv("hsigU", hsigU); kv("rd", rd);
            karr("fg", fgs, n);
            karr("P_norm", pn, n);
            // genulens divides by its trapezoid integral over [fgmin, fgmax]
            const int jmid = n / 2;
            kv("norm", calc_PRRg(ctx, Rn, zn, fgs[jmid], sigU0, hsigU, rd) / pn[jmid]);
            std::vector<double> fgrid, pgrid;
            for (int j = 1; j <= 300; ++j) {
                const double fg = 0.01 * j;
                fgrid.push_back(fg);
                pgrid.push_back(calc_PRRg(ctx, Rn, zn, fg, sigU0, hsigU, rd));
            }
            karr("fg_grid", fgrid);
            karr("P_unnorm_grid", pgrid);
            close_obj();
        }
    }
    close_arr();

    // ---- mass: the IMF grid, turnoff tables, the IFMR -------------------------
    const auto &s = ctx.stellar;
    key("mass");
    open_obj();
    kv("logMst", s.logMst); kv("dlogM", s.dlogM); kv("nm", s.nm);
    karr("PlogM", pop.mass_probability, s.nm + 1);
    karr("agesD", s.agesD.data(), s.nageD); karr("MinidieD", s.MinidieD.data(), s.nageD);
    karr("agesB", s.agesB.data(), s.nageB); karr("MinidieB", s.MinidieB.data(), s.nageB);
    karr("agesND", s.agesND.data(), s.nageND); karr("MinidieND", s.MinidieND.data(), s.nageND);
    kv("MiniWDmax", s.MiniWDmax);
    std::vector<double> mini, mmean, tmean;
    for (int j = 0; j <= 400; ++j) {
        const double mi = std::pow(10.0, std::log10(0.5) + j * (std::log10(120.0) - std::log10(0.5)) / 400);
        double pout[2] = {};
        Mini2Mrem(ctx, pout, mi, 1);
        mini.push_back(mi); mmean.push_back(pout[0]); tmean.push_back(pout[1]);
    }
    karr("Mini", mini); karr("Mrem_mean", mmean); karr("type_mean", tmean);
    // the random branch: remnant-mass quantiles and type fractions per Mini
    key("Mrem_mc");
    open_arr();
    for (double mi : {9.5, 12.0, 14.0, 16.0, 18.0, 20.0, 23.0, 26.0, 30.0, 50.0, 70.0, 100.0}) {
        std::vector<double> ms;
        int nns = 0;
        for (int j = 0; j < nmc; ++j) {
            double pout[2] = {};
            Mini2Mrem(ctx, pout, mi, 0);
            ms.push_back(pout[0]);
            if (pout[1] == 2.0) ++nns;
        }
        sep(); open_obj();
        kv("Mini", mi); kv("f_ns", static_cast<double>(nns) / nmc);
        karr("q", quantiles(ms, probs));
        close_obj();
    }
    close_arr();
    close_obj();

    // ---- the luminosity functions (only with an AIrc source selection) ------
    if (pop.has_luminosity_function) {
        key("lf");
        open_obj();
        karr("MIs", ctx.luminosity.MIs, ctx.luminosity.nMIs);
        key("CumuN_MIs");
        open_arr();
        for (int i = 0; i < 11; ++i) {
            sep(); open_arr();
            for (int j = 0; j < ctx.luminosity.nMIs; ++j) num(ctx.luminosity.CumuN_MIs[i][j]);
            close_arr();
        }
        close_arr();
        close_obj();
    }
    close_obj();
    std::printf("\n");
    return 0;
}
