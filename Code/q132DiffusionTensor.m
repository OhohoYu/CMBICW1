
clear all;
close all;
% Load the diffusion signal %
fid = fopen('isbi2015_data_normalised.txt', 'r', 'b'); fgetl(fid); % Read in the header
D = fscanf(fid, '%f', [6, inf])'; % Read in the data fclose(fid);
% Select the first of the 6 voxels
voxel = D(:,1);
% Load the protocol %
fid = fopen('isbi2015_protocol.txt', 'r', 'b'); fgetl(fid);
A = fscanf(fid, '%f', [7, inf]);
fclose(fid);
% Create the protocol
qhat = A(1:3,:); G = A(4,:);
delta = A(5,:); smalldel = A(6,:); TE = A(7,:);
GAMMA = 2.675987E8;
bvals = ((GAMMA*smalldel.*G).^2).*(delta-smalldel/3); % convert bvals units from s/m^2 to s/mm^2
bvals = bvals/10^6;

initial_point = [3.5e+0 3e-03 2.5e-01 0 0];



% Define various options for the non­linear fitting algorithm.
h=optimset('MaxFunEvals',20000,...
   'Algorithm','quasi-newton',...
   'TolX',1e-10,...
   'Display','none',...
   'TolFun',1e-10);

% Now run the fitting

[new_res,predicted_y] = DiffusionTensor(voxel,qhat,bvals);
disp(new_res);
newGraph = plotFit(predicted_y, voxel);
saveas(newGraph, 'Diagrams/q132-DiffusionModel-voxel1.png');

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

function [sumRes, resJ] = DiffusionTensor(Avox,qhat,bvals)
%   S = exp(y.x) ->   x = log(S)/y
    [q_x, q_y, q_z] = deal(qhat(1,:), qhat(2,:), qhat(3,:));
    
%   y = [1 ?bqx2 -2bqxqy -2bqxqz ?bqy2 -2bqyqz ?bqz2]   
    y1 = ones(1,length(Avox));
    y2 = - bvals .* q_x .^ 2;
    y3 = -2 * bvals .* q_x .* q_y;
    y4 = -2 * bvals .* q_x .* q_z;
    y5 = -bvals .* q_y .^2;
    y6 = -2 * bvals .* q_y .* q_z; 
    y7 = -bvals .* q_z .^ 2;
    y = [y1;y2;y3;y4;y5;y6;y7];
    y_t = y';
    
%   x = [log S(0,0) Dxx Dxy Dxz Dyy Dyz Dzz ]  

    x = y_t \ log(Avox);
    S = exp(y_t * x);
    sumRes = sum((Avox-S).^2);
    resJ = S;

end

