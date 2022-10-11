%% Import Script for EBSD Data
%
% This script was automatically created by the import wizard. You should
% run the whoole script or parts of it in order to import your data. There
% is no problem in making any changes to this script.

%% Specify Crystal and Specimen Symmetries

% crystal symmetry
CS = {... 
  'notIndexed',...
  crystalSymmetry('-3m1', [68 68 141], 'X||a*', 'Y||b', 'Z||c*', 'mineral', 'Material1', 'color', [0.53 0.81 0.98])};

% plotting convention
setMTEXpref('xAxisDirection','north');
setMTEXpref('zAxisDirection','outOfPlane');

%% Specify File Names

% path to files
pname = '/home/esrf/purushot/Desktop';

% which files to be imported
fname = [pname '/ZnCuOCl_MTEX_UBmat_0abfree.ctf'];

%% Import the Data

% create an EBSD variable containing the data
ebsd = EBSD.load(fname,CS,'interface','ctf',...
  'convertEuler2SpatialReferenceFrame');

[grains,ebsd.grainId,ebsd.mis2mean] = calcGrains(ebsd('indexed'),'threshold',5*degree);
%% HKL colors
mat_key = "Material1";
mat_keys = "ZnCuOCl_abfree";
ipfaztecKey = ipfHKLKey(ebsd(mat_key));
ipfaztecKey.inversePoleFigureDirection = vector3d.Z;
colors = ipfaztecKey.orientation2color(ebsd(mat_key).orientations);

%colorkey
f7 = figure;
plot(ipfaztecKey);
savenm = sprintf('colorkey_%s.png',mat_keys);
saveas(f7, fullfile(pname,savenm));

%EBSD
f7 = figure;
plot(ebsd(mat_key),colors, 'micronbar','off');
savenm = sprintf('ebsd_%s.png',mat_keys);
saveas(f7, fullfile(pname,savenm));

%misorientation
%intragranular
mori = ebsd(mat_key).mis2mean;

f7 = figure;
plotAngleDistribution(mori);
xlabel('Misorientation angles in deg')
savenm = sprintf('mori_distribution_%s.png',mat_keys);
saveas(f7, fullfile(pname,savenm));

f7 = figure;
plot(ebsd(mat_key),ebsd(mat_key).mis2mean.angle./degree, 'micronbar','off');
mtexColorMap hot
mtexColorbar
savenm = sprintf('mori_map_%s.png',mat_keys);
saveas(f7, fullfile(pname,savenm));

%GOS
f7 = figure;
plot(grains(mat_key),grains(mat_key).GOS./degree,'micronbar','off')
mtexColorbar
savenm = sprintf('GOS_map_%s.png',mat_keys);
saveas(f7, fullfile(pname,savenm));

%KAM
f7 = figure;
KAM = ebsd(mat_key).KAM;
plot(ebsd(mat_key),KAM);
mtexColorbar
savenm = sprintf('KAM_map_%s.png',mat_keys);
saveas(f7, fullfile(pname,savenm));

%GAM
f7 = figure;
GAM = accumarray(ebsd(mat_key).grainId, KAM, size(grains), @nanmean) ./degree;
plot(grains(mat_key),GAM(grains(mat_key).id),'micronbar','off')
mtexColorbar
savenm = sprintf('GAM_map_%s.png',mat_keys);
saveas(f7, fullfile(pname,savenm));

%PF 
h = Miller({0,0,0,1},{-1,1,0,0},{0,1,-1,0},ebsd(mat_key).CS);

f7 = figure;
plotPDF(ebsd(mat_key).orientations,colors,h,'MarkerSize',5,...
  'MarkerFaceAlpha',0.05,'MarkerEdgeAlpha',0.05, 'all');
savenm = sprintf('pf_%s.png',mat_keys);
saveas(f7, fullfile(pname,savenm));

%IPF SST
r = vector3d.Z;
f7 = figure;
plotIPDF(ebsd(mat_key).orientations,colors,r,'MarkerSize',5,...
  'MarkerFaceAlpha',0.05,'MarkerEdgeAlpha',0.05, 'all');
savenm = sprintf('ipf_%s.png',mat_keys);
saveas(f7, fullfile(pname,savenm));