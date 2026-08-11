/// <reference types="vite/client" />

// plotly.js-dist-min ships no bundled types and we do not pull @types for it
// (the full @types/plotly.js is large and pulls d3 typings). We use a narrow
// structural shim covering only the calls the adapter makes.
declare module "plotly.js-dist-min" {
  export function newPlot(
    root: HTMLElement,
    data: unknown[],
    layout?: unknown,
    config?: unknown
  ): Promise<void>;
  export function react(
    root: HTMLElement,
    data: unknown[],
    layout?: unknown,
    config?: unknown
  ): Promise<void>;
  export function purge(root: HTMLElement): void;
  export function Plots(): void;
  const Plotly: {
    newPlot: typeof newPlot;
    react: typeof react;
    purge: typeof purge;
  };
  export default Plotly;
}

// Static asset imports resolved by Vite.
declare module "*.svg" {
  const src: string;
  export default src;
}
declare module "*.png" {
  const src: string;
  export default src;
}
