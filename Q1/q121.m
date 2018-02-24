
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
[end_params, RESNORM] = findGlobalMin(x0,Avox,bvals,qhat,h);
[diff, predicted_y] = BallStickSSD2(end_params,Avox,bvals,qhat);
std_y = sqrt(diff/103);


num_iterations = 1000;
parameter_dists = zeros(num_iterations,5);

for i=1:num_iterations
    modified_predicted = predicted_y + normrnd(0,std_y, size(predicted_y));
    [new_param,~,~,~] = fminunc(@BallStickSSD2, end_params, h, modified_predicted', bvals,qhat);
    [p1,p2,p3,p4,p5] = newTransform(new_param);
    parameter_dists(i,:) = deal([p1,p2,p3,p4,p5]);
end
save('q121.mat','parameter_dists');

load('q121.mat', 'parameter_dists');

for i=1:size(parameter_dists(1,:),2)
    mean_params = mean(parameter_dists(:,i));
    std_params = std(parameter_dists(:,i));
    length_params = length(parameter_dists(:,i));
    filename = sprintf('Diagrams/Q2/q121-parameter%d.png', i);

    sorted_params = sort(parameter_dists(:,i));
    % Getting the upper and lower limits for 95% confidence (2.5% confidence on
    % either side)
    [l_lim, u_lim] = deal(sorted_params(floor(0.025*length_params)),sorted_params(ceil(0.975*length_params))); 
    % Sample standard deviation
    sample_std = [mean_params - std_params, mean_params + std_params];
    confidence = [l_lim, u_lim];
    h = figure;
    hist1 = histfit(sorted_params,50);
    y_bound = ylim;
    hold on;
    set(hist1(1),'facecolor','none'); set(hist1(2),'color','m');
    plot(confidence, 0.5*[y_bound(2) ,y_bound(2)],'g-x', 'LineWidth',0.75);
    plot(sample_std, 0.6*[y_bound(2) ,y_bound(2)],'b-x', 'LineWidth', 0.75);
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
    finalParams = global_params;
    resnorm = global_min;
end
