
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
   'Algorithm','quasi-newton',...
   'TolX',1e-10,...
   'Display','none',...
   'TolFun',1e-10);

% Define a starting point for the non­linear fit
x0 = getInformedStart(Avox,qhat,bvals);
[end_params, RESNORM] = findGlobalMin(x0,Avox,bvals,qhat,h);


[diff, predicted_y] = BallStickSSD(end_params,Avox,bvals,qhat);
std_y = sqrt((diff)/(103));

[acceptance, parameter_samples, ratios, xs] = computeMCMC(end_params,Avox,bvals,qhat,diff);
for j=1:3
    f = figure;
    plot(parameter_samples(j,10000:end));
    filename = sprintf('Diagrams/Q2/q122-parameterPlots%d.png', j);
    saveas(f,filename);
end

load('q121.mat','parameter_dists');

for i=1:3
    mean_params = mean(parameter_dists(:,i));
    std_params = std(parameter_dists(:,i));
    length_params = length(parameter_dists(:,i));
    filename = sprintf('Diagrams/Q2/q122-parameterLims%d.png', i);
    sorted_params = sort(parameter_dists(:,i));
    % Getting the upper and lower limits for 95% confidence (2.5% confidence on
    % either side)
    [l_lim, u_lim] = deal(sorted_params(floor(0.025*length_params)),sorted_params(ceil(0.975*length_params))); 
    
    % Sample standard deviation
    sample_std = [mean_params - std_params, mean_params + std_params];
    sample_confidence = [l_lim, u_lim];
    
    mean_mcmc = mean(parameter_samples(i,10000:end));
    std_mcmc = std(parameter_samples(i,10000:end));
    sorted_mcmc = sort(parameter_samples(i,10000:end));
    [l_lim_mcmc, u_lim_mcmc] = deal(sorted_mcmc(floor(0.025*length(sorted_mcmc))),sorted_mcmc(ceil(0.975*length(sorted_mcmc)))); 
    
    mcmc_std= [mean_mcmc - std_mcmc, mean_mcmc + std_mcmc];
    mcmc_confidence = [l_lim_mcmc, u_lim_mcmc];
    h = figure;
    plot(mcmc_confidence, [4,4],'r--x', 'LineWidth', 0.75);
    ylim([0 5]);
    hold on;
    plot(mcmc_std, [3,3],'k--x', 'LineWidth', 0.75);
    plot(sample_confidence, [2,2],'g:x', 'LineWidth',0.75);
    plot(sample_std, [1,1],'b:x', 'LineWidth', 0.75);
    legend('Confidence limits MCMC','Standard Deviation MCMC','Confidence limits Parametric','Standard Deviation Parametric','location','northoutside');
    hold off;
    saveas(h,filename);
    
    filename = sprintf('Diagrams/Q2/q122-parameterHist%d.png', i);
    clf(h,'reset');
    hist1 = histfit(sorted_mcmc,50);
    y_bound = ylim;
    hold on;
    set(hist1(1),'facecolor','none'); set(hist1(2),'color','m');
    plot(mcmc_confidence, 0.5*[y_bound(2) ,y_bound(2)],'g-x', 'LineWidth',0.75);
    plot(mcmc_std, 0.6*[y_bound(2) ,y_bound(2)],'b-x', 'LineWidth', 0.75);
    legend('Likelihood distribution (histogram)','Likelihood Fit (histfit)','95% Range','Standard Deviation Range','location','northoutside');
    saveas(h,filename);
    
end


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

function [finalParams, resnorm] = findGlobalMin(x0, Avox, bvals, qhat,h)

    [x1,x2,x3,x4,x5] = newInverse(x0);
    x_inv = [x1, x2, x3, x4, x5];
    
    % Now run the fitting
    [parameter_hat,RESNORM,~,~] = fminunc(@BallStickSSD2, x_inv, h, Avox, bvals,qhat);
    global_min = RESNORM;

    num_iterations = 100;
    counter = 0;
    tic;
    normalised_start = x0 + 0.0001;
    global_set = ones(1,num_iterations);
    global_params = parameter_hat;
    for i=1:num_iterations
        random_modifier = rand(size(normalised_start));

        [z1,z2,z3,z4,z5] = newInverse((normalised_start) .* random_modifier);
        random_start = [z1,z2,z3,z4,z5];
        [parameter_hat,new_res,~,~] = fminunc(@BallStickSSD2, random_start, h, Avox, bvals,qhat);
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
    [a1,a2,a3,a4,a5] = newTransform(global_params);
    finalParams = [a1,a2,a3,a4,a5];
    resnorm = global_min;
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
    eig_2 = sorted_eig(2);
    [~, positions] = max(elements);
    direction = eigenV(:,positions);
%   Convert cartesian to spherical coordinates
    
    [phi, theta] = cart2sph(direction(1),direction(2),direction(3));
    x = [S0,diff,f,theta,phi,eig_1,eig_2];
    
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

function [acceptance_rate, parameter_samples, ratios, new_xs] = computeMCMC(x0,Avox,bvals,qhat,initial_resnorm)
    num_iterations = 100000;
    pertubation_steps = [10000,0.0002,0.01,0.1,0.1];    
    burn_in = num_iterations / 10; 
    new_params = ones(5, num_iterations+burn_in);
    x_old = x0;
    res_old = initial_resnorm;
    tic;
    counter = 0;
    ratios = zeros(1,num_iterations+burn_in);
    new_xs = zeros(5,num_iterations+burn_in);
    for i=1:(num_iterations + burn_in)
        x_new = normrnd(x_old, pertubation_steps, size(x0));
        new_xs(:,i) = x_new; 
        [res_new, ~] = BallStickSSD(x_new,Avox,bvals,qhat);
        acceptance_ratio = ((res_old-res_new) / (2*10000^2));
        ratios(i) = acceptance_ratio;
        if (acceptance_ratio > log(rand))
            new_params(:,i) = x_new;
            counter = counter + 1;
            x_old = x_new;
            res_old = res_new;
        else
            new_params(:,i) = x_old;
        end
    end
    toc;
    acceptance_rate = counter / (num_iterations + burn_in);
    parameter_samples = new_params;
    
    save('q122.mat','new_params');


end