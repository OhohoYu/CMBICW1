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

% Define various options for the non­linear fitting algorithm.
h=optimset('MaxFunEvals',20000,...
   'Algorithm','levenberg-marquardt',...
   'TolX',1e-10,...
   'Display','iter',...
   'TolFun',1e-10);

% Now run the fitting
[parameter_hat,RESNORM,EXITFLAG,OUTPUT] = fminunc(@BallStickSSD, x0, h, Avox, bvals,qhat);

[diff, predicted_y] = BallStickSSD(parameter_hat,Avox,bvals,qhat);
newGraph = plotFit(predicted_y, Avox);
saveas(newGraph, 'q111.png');


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

function [sumRes, resJ] = BallStickSSD(x0, Avox, bvals,qhat)
    % Extract the parameters
    S0 = x0(1);
    diff = x0(2);
    f = x0(3);
    theta = x0(4);
    phi = x0(5);
    
    % Synthesize the signals
    fibdir = [cos(phi)*sin(theta) sin(phi)*sin(theta) cos(theta)];
    fibdotgrad = sum(qhat.*repmat(fibdir, [length(qhat) 1])');
    S = S0*(f*exp(-bvals*diff.*(fibdotgrad.^2)) + (1-f)*exp(-bvals*diff));
    
    % Compute the sum of square differences
    sumRes = sum((Avox - S').^2);
    resJ = S;
end








