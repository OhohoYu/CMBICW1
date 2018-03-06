
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

load('q131.mat');

fisher = calculateFisher(final_params, voxel, qhat, bvals);
disp(fisher);

function [fisher] = calculateFisher(params, Avox, qhat, bvals)
    length_params = length(params);
    length_input = length(Avox);
    fisher = zeros(length_params,length_params);
    derivs = getDerivatives(params,bvals,qhat);
    for i=1:length_input
        fisher = derivs(i,:)'*derivs(i,:);
    end
    fisher = fisher./(0.04^2);
    fisher = fisher./(params'*params);
end

function [derivs] = getDerivatives(x0, bvals,qhat)
    % Extract the parameters
    S0 = x0(1);
    diff = x0(2);
    f = x0(3);
    theta = x0(4);
    phi = x0(5);
    
    fibdir = [cos(phi)*sin(theta) sin(phi)*sin(theta) cos(theta)];
    fibdotgrad = sum(qhat.*repmat(fibdir, [length(qhat) 1])');    
    S_i = exp(-bvals * diff .* (fibdotgrad.^2));
    S_e = exp(-bvals*diff);
    deriv_F = S0*(S_i-S_e);
    
    intermediate = S0*f*S_i .* (-(2*bvals*diff) .* fibdotgrad);
    ddTheta = repmat([cos(phi)*cos(theta),sin(phi)*cos(theta),-sin(theta)], [length(qhat) 1]);
    ddPhi = repmat([-sin(phi)*sin(theta),cos(phi)*sin(theta),0], [length(qhat) 1]);
    deriv_Phi = intermediate.*sum(ddPhi'.*qhat);
    deriv_Theta = intermediate.*sum(ddTheta'.*qhat);
    
    deriv_Diff = S0*(f*S_i.*(-bvals*diff.*(fibdotgrad.^2))) + ((1-f)*S_e.*(-bvals));
    deriv_S0 = f*S_i + (1-f)*S_e;
    derivs = [deriv_S0;deriv_Diff;deriv_F;deriv_Theta;deriv_Phi]';
end
