

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



% Define various options for the non­linear fitting algorithm.
h=optimset('MaxFunEvals',20000,...
   'Algorithm','quasi-newton',...
   'TolX',1e-10,...
   'Display','none',...
   'TolFun',1e-10);

% Now run the fitting
[initial_point] = getInformedStart(voxel,qhat,bvals);
% [parameter_hat,RESNORM,~,~] = fminunc(@ZeppelinStick, initial_point, h, voxel, bvals,qhat);
[final_params,new_res] = findGlobalMin(initial_point,voxel,qhat,bvals,h);
[new_res,predicted_y] = ZeppelinStickTortuosity(final_params, voxel,bvals,qhat);

% [new_res,predicted_y] = findGlobalMin(initial_point, voxel,qhat,bvals,h);
save('q132ZeppTort.mat','final_params','new_res');
newGraph = plotFit(predicted_y, voxel);
saveas(newGraph, 'Diagrams/q132-ZeppelinStickTortuositySSD-voxel1.png');

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

function [sumRes, resJ] = ZeppelinStickTortuosity(x0, Avox, bvals,qhat)
    % Extract the parameters
    S0 = x0(1);
    diff = x0(2);
    f = x0(3);
    theta = x0(4);
    phi = x0(5);
    eig_1 = x0(6);
    eig_2 = (1-f)*eig_1;
    
    % Synthesize the signals
    fibdir = [cos(phi)*sin(theta) sin(phi)*sin(theta) cos(theta)];
    fibdotgrad = sum(qhat.*repmat(fibdir, [length(qhat) 1])');
    S_i = exp(-bvals*diff.*(fibdotgrad.^2));
    S_e = exp(-bvals.*(eig_2 + (eig_1 - eig_2).*(fibdotgrad.^2)));
    S = S0*(f*S_i + (1-f)*S_e);
    
    % Compute the sum of square differences
    sumRes = sum((Avox - S').^2);
    resJ = S;
end

function [x] =  getInformedStart(Avox, qhat, bvals)
    [~,~,S0, dMatrix] = DiffusionTensor(Avox,qhat,bvals);
    
    diff = mean(dMatrix(:));
    
    [eigenV,eigenD] = eig(dMatrix);
    
    normalised_eig = eigenD./sum(eigenD);
    f1 = abs(normalised_eig(1) - normalised_eig(2));
    f2 = abs(normalised_eig(2) - normalised_eig(3));
    f3 = abs(normalised_eig(3) - normalised_eig(1));
    f = (1/2)  * (f1 + f2 + f3);
    
    elements = diag(eigenD);
    sorted_eig = sort(elements);
    eig_1 = sorted_eig(3);
    [~, positions] = max(elements);
    direction = eigenV(:,positions);
%   Convert cartesian to spherical coordinates
    
    [phi, theta] = cart2sph(direction(1),direction(2),direction(3));
    x = [S0,diff,f,theta,phi,eig_1];
    
end

function [sumRes, resJ, S0, dMatrix] = DiffusionTensor(Avox,qhat,bvals)
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
    
    x = y_t \ log(Avox);
    S0 = exp(x(1));
    dMatrix = zeros(3,3);
    dMatrix(1,:) = [x(2),x(3),x(4)];
    dMatrix(2,:) = [x(3),x(5),x(6)];
    dMatrix(3,:) = [x(4),x(6),x(7)];
    
    S = exp(y_t * x);
    sumRes = sum((Avox-S').^2);
    resJ = S;
    

end


function [finalParams, resnorm] = findGlobalMin(x0, Avox, qhat, bvals,h)
    
    % Now run the fitting
    [parameter_hat,RESNORM,~,~] = fminunc(@ZeppelinStickTortuosity, x0, h, Avox, bvals,qhat);
    global_min = RESNORM;

    num_iterations = 200;
    counter = 0;
    tic;
%     normalised_start = x0 + 0.0000000000001;
    global_set = ones(1,num_iterations);
    global_params = parameter_hat;
    for i=1:num_iterations
        state = 1;
        while(state == 1)
            try
                random_start = normrnd(x0, [0.15,1e-09,0.1,2*pi,2*pi,1e-09]);
%               random_start = normalised_start .* random_modifier;
                [parameter_hat,new_res,~,~] = fminunc(@ZeppelinStickTortuosity, random_start, h, Avox, bvals,qhat);
                state = 0;
            catch
            end
        end        
        global_set(i) = new_res;
        if (abs(new_res - global_min) <= 0.1)
            counter = counter + 1;
        elseif ((global_min - new_res) > 0.1)
            global_min = new_res;
            counter = 0;
            global_params = parameter_hat;
        end
    end
    toc;
    disp(global_min);
    disp(counter);
    finalParams = global_params;
    resnorm = global_min;
end
