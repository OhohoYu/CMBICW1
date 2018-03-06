clear all;
close all;

load('q131.mat','final_params','new_res');
numPoints = 3612;

BallStickParams = final_params;
BallStickRESNORM = new_res;

load('q132Zepp.mat','final_params','new_res');
ZeppStickParams = final_params;
ZeppStickRESNORM = new_res;

load('q132ZeppTort.mat','final_params','new_res');
ZeppTortParams = final_params;
ZeppTortRESNORM = new_res;

aicBallStick = AIC(BallStickRESNORM,5);
aicZeppStick = AIC(ZeppStickRESNORM,7);
aicZeppTort = AIC(ZeppTortRESNORM,6);

bicBallStick = BIC(numPoints,BallStickRESNORM,5);
bicZeppStick = BIC(numPoints,ZeppStickRESNORM,7);
bicZeppTort = BIC(numPoints,ZeppTortRESNORM,6);

disp("AIC");
disp(aicBallStick);
disp(aicZeppStick);
disp(aicZeppTort);

disp("BIC");
disp(bicBallStick);
disp(bicZeppStick);
disp(bicZeppTort);


function [result] = AIC(ssd, numParams)
    result = (2*numParams) + (ssd/(0.04^2));
end

function [result] = BIC(numPoints, ssd, numParams)
    result = (numParams*log(numPoints)) + (ssd/(0.04^2));
end

