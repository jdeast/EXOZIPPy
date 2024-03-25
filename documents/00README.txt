See also: https://github.com/jdeast/EXOFASTv2/blob/master/exofastv2.pro
# Conventions
## Filenames

File Formatting convention: nYYYYMMDD.filtername.telescope.whateveryouwant.
YYYYMMDD = (arbitrary, used as a label) Suggest UT date of first datapoint or
beginning of relevant microlensing season.

## Filternames
Johnson/Cousins: 'U','B','V','R','I','J','H','K'
Sloan: 'Sloanu','Sloang','Sloanr','Sloani','Sloanz'
Kepler: 'Kepler'
CoRoT: 'CoRoT'
TESS: 'TESS'
Spitzer: 'Spit36','Spit45','Spit58','Spit80'
Stromgren: 'u','b','v','y'

## Output
### .csv
varname median lower68% upper68%

multi-modal: separate csv file or separate section in a single file
for each solution.
