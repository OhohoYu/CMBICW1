
clear all;
close all;
load('data');
dwis=double(dwis);
dwis=permute(dwis,[4,1,2,3]);

qhat = load('bvecs');
bvals = 1000*sum(qhat.*qhat);

Avox = dwis(:,92,65,72);

% Define various options for the non­linear fitting algorithm.
h=optimset('MaxFunEvals',20000,...
   'Algorithm','levenberg-marquardt',...
   'TolX',1e-10,...
   'Display','iter',...
   'TolFun',1e-10);

% Now run the fitting for x iteration

iterations = 120;
global_min = zeros(iterations, 1);
for i = 1:iterations
    max_points = [3.5e+0 3e-03 2.5e-01 0 0];
    random_number = randn(size(max_points));
    [x1,x2,x3,x4,x5] = newTransform((max_points+0.001).*random_number);
    transformed = [x1,x2,x3,x4,x5];
    start_x = transformed; 
    
    [parameter_hat,RESNORM,EXITFLAG,OUTPUT] = fminunc(@BallStickSSD2, start_x, h, Avox, bvals,qhat);
    global_min(i) = RESNORM;
end
global_min = sort(global_min);
min(global_min)

function [x1, x2, x3, x4, x5] = newTransform(x)
    x1 = x(1)^2;
    x2 = x(2)^2;
    x3 = 1/(1+exp(-1*x(3)));
    x4 = x(4);
    x5 = x(5);
end

function [sumRes, resJ] = BallStickSSD2(x0, Avox, bvals,qhat)
    [ x1, x2, x3, x4, x5] = newTransform(x0);
    % Extract the parameters
    S0 = x1;
    diff = x2;
    f = x3;
    theta = x4;
    phi = x5;
    
    % Synthesize the signals
    fibdir = [cos(phi)*sin(theta) sin(phi)*sin(theta) cos(theta)];
    fibdotgrad = sum(qhat.*repmat(fibdir, [length(qhat) 1])');
    S = S0*(f*exp(-bvals*diff.*(fibdotgrad.^2)) + (1-f)*exp(-bvals*diff));
    
    % Compute the sum of square differences
    sumRes = sum((Avox - S').^2);
    resJ = S;
end   







