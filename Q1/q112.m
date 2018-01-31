
clear all;
close all;
load('data');
dwis=double(dwis);
dwis=permute(dwis,[4,1,2,3]);

qhat = load('bvecs');
bvals = 1000*sum(qhat.*qhat);

Avox = dwis(:,92,65,72);

% Define a starting point for the non­linear fit
x0 = [3.5e+0 3e-03 2.5e-01 0 0];
[x1,x2,x3,x4,x5] = newInverse(x0);
x_inv = [x1, x2, x3, x4, x5];

% Define various options for the non­linear fitting algorithm.
h=optimset('MaxFunEvals',20000,...
   'Algorithm','levenberg-marquardt',...
   'TolX',1e-10,...
   'Display','iter',...
   'TolFun',1e-10);

% Now run the fitting
[parameter_hat,RESNORM,EXITFLAG,OUTPUT] = fminunc(@BallStickSSD2, x_inv, h, Avox, bvals,qhat);
[diff, predicted_y] = BallStickSSD2(parameter_hat,Avox,bvals,qhat);
newGraph = plotFit(predicted_y, Avox);
saveas(newGraph, 'q112.png');

function [x1, x2, x3, x4, x5] = newInverse(x)
    x1 = sqrt(x(1));
    x2 = sqrt(x(2));
    x3 = -log((1/x(3)) - 1);
    x4 = x(4);
    x5 = x(5);
end

function [x1, x2, x3, x4, x5] = newTransform(x)
    x1 = x(1)^2;
    x2 = x(2)^2;
    x3 = 1/(1+exp(-1*x(3)));
    x4 = x(4);
    x5 = x(5);
end

function graph = plotFit(predicted_y, Avox)
    graph = figure();
    % Plot the actual data points
    plot(Avox, ' go', 'MarkerSize', 4, 'LineWidth', 4)
    hold on;
    % Add the predictions to the plot
    plot(predicted_y,' bd', 'MarkerSize', 4, 'LineWidth', 4)
    xlabel('\bf{q} index');
    ylabel('S');
    legend('Data', 'Model');

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








