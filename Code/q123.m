
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
x0 = [3.5e+0, 3e-03, 2.5e-01, 0, 0];
[end_params, RESNORM, hessian] = findGlobalMin(x0,Avox,bvals,qhat,h);
[diff, predicted_y] = BallStickSSD2(end_params,Avox,bvals,qhat);
std_y = sqrt((diff)/(103));
cov = -inv(hessian/(-10000^2));

sigma = sqrt(diag(cov));

twoSigma = [(end_params' -sigma), (end_params' + sigma)];
laplace_std = twoSigma(1:3,:);

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
    filename = sprintf('Diagrams/Q2/q123-parameterLims%d.png', i);
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
    plot(mcmc_confidence, [5,5],'r--x', 'LineWidth', 0.75);
    ylim([0 6]);
    hold on;
    plot(mcmc_std, [4,4],'k--x', 'LineWidth', 0.75);
    plot(sample_confidence, [3,3],'g:x', 'LineWidth',0.75);
    plot(sample_std, [2,2],'b:x', 'LineWidth', 0.75);
    plot(laplace_std(i,:), [1,1],'c-o', 'LineWidth', 0.75);
    legend('Confidence limits MCMC','Standard Deviation MCMC','Confidence limits Parametric','Standard Deviation Parametric','Laplace','location','northoutside');
    hold off;
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

function [finalParams, resnorm, hessian] = findGlobalMin(x0, Avox, bvals, qhat,h)

    [x1,x2,x3,x4,x5] = newInverse(x0);
    x_inv = [x1, x2, x3, x4, x5];
    
    % Now run the fitting
    [parameter_hat,RESNORM,~,~,~,hessian] = fminunc(@BallStickSSD2, x_inv, h, Avox, bvals,qhat);
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
        [parameter_hat,new_res,~,~,~,new_hessian] = fminunc(@BallStickSSD2, random_start, h, Avox, bvals,qhat);
        global_set(i) = new_res;
        if (abs(new_res - global_min) <= 0.1)
            counter = counter + 1;
        elseif ((global_min - new_res) > 0.1)
            global_min = new_res;
            counter = 0;
            global_params = parameter_hat;
            hessian = new_hessian;
        end
    end
    toc;
    disp(global_min);
    disp(counter);
    finalParams = global_params;
    resnorm = global_min;
end

function [acceptance_rate, parameter_samples, ratios, new_xs] = computeMCMC(x0,Avox,bvals,qhat,initial_resnorm)
    num_iterations = 100000;
%     covariance = eye(5);
%     covariance(1,1) = 100;
%     covariance(2,2) = 0.00005;
%     covariance(3,3) = 0.01;
%     covariance(4,4) = 0.1;
%     covariance(5,5) = 0.1;
%     covariance = covariance.^2;
    
    pertubation_steps = [100,0.00005,0.01,0.1,0.1];    
    burn_in = num_iterations / 10; 
    new_params = ones(5, num_iterations+burn_in);
    x_old = x0;
    res_old = initial_resnorm;
    tic;
    counter = 0;
    ratios = zeros(1,num_iterations+burn_in);
    new_xs = ones(5,num_iterations+burn_in);
    for i=1:(num_iterations + burn_in)
        x_new = normrnd(x_old, pertubation_steps, size(x0));
        new_xs(:,i) = x_new; 
        [z1,z2,z3,z4,z5] = newInverse(x_new);
        inv_new = [z1,z2,z3,z4,z5];
        [res_new, ~] = BallStickSSD2(inv_new,Avox,bvals,qhat);
        acceptance_ratio = ((res_old-res_new) / (10000^2));
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
    
    save('q123.mat','new_params');


end