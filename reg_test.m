close all
clear

moving = pcread("object.ply");
fixed = pcread("scene.ply");
fixed = pcdenoise(fixed);

figure
pcshowpair(moving,fixed,VerticalAxis="Y",VerticalAxisDir="Down") 

fixedDownsampled = pcdownsample(fixed,gridAverage=0.01);
movingDownsampled = pcdownsample(moving,gridAverage=0.01);

figure
pcshowpair(movingDownsampled,fixedDownsampled,VerticalAxis="Y",VerticalAxisDir="Down")

gridSize = 0.1;
T = pcregisterfgr(movingDownsampled,fixedDownsampled, gridSize);

movingRegistered = pctransform(moving,T);
figure
pcshowpair(movingRegistered,fixed,VerticalAxis="Y",VerticalAxisDir="Down");
