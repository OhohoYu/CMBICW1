
clear all;
close all;
load('data');
dwis=double(dwis);
dwis=permute(dwis,[4,1,2,3]);

qhat = load('bvecs');
bvals = 1000*sum(qhat.*qhat);

Avox = dwis(:,55,39,48);

% Define a starting point for the non­linear fit
x0 = getInformedStart(Avox,qhat,bvals);
[x1,x2,x3,x4,x5] = newInverse(x0);
x_inv = [x1, x2, x3, x4, x5];

% Define various options for the non­linear fitting algorithm.
h=optimset('MaxFunEvals',20000,...
   'Algorithm','quasi-newton',...
   'TolX',1e-10,...
   'Display','none',...
   'TolFun',1e-10);


% Now run the fitting
[parameter_hat,RESNORM,EXITFLAG,OUTPUT] = fminunc(@BallStickSSD2, x_inv, h, Avox, bvals,qhat);
[diff, predicted_y] = BallStickSSD2(parameter_hat,Avox,bvals,qhat);
global_min = RESNORM;

numIterations = 200;
counter = 0;
tic;
normalised_start = x0 + 0.0001;
global_set = ones(1,numIterations);
for i=1:numIterations
    random_modifier = rand(size(normalised_start));
    [z1,z2,z3,z4,z5] = newInverse((normalised_start) .* random_modifier);
    random_start = [z1,z2,z3,z4,z5];
    [parameter_hat,new_res,EXITFLAG,OUTPUT] = fminunc(@BallStickSSD2, random_start, h, Avox, bvals,qhat);
    global_set(i) = new_res;
    if (abs(new_res - global_min) <= 0.1)
        counter = counter + 1;
    elseif ((global_min - new_res) > 0.1)
        global_min = new_res;
        counter = 0;
    end
end
toc;
disp(global_min);
disp(counter);

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

function [x] =  getInformedStart(Avox, qhat, bvals)
    x = zeros(1,5);
    [~,~,S0, dMatrix] = DiffusionTensor(Avox,qhat,bvals);
    
    D = mean(dMatrix(:));
    
    [eigenV,eigenD] = eig(dMatrix);
    normalised_eig = eigenD./sum(eigenD);
    f1 = abs(normalised_eig(1) - normalised_eig(2));
    f2 = abs(normalised_eig(2) - normalised_eig(3));
    f3 = abs(normalised_eig(3) - normalised_eig(1));
    f = (1/2)  * (f1 + f2 + f3);
    
    elements = diag(eigenD);
    [~, positions] = max(elements);
    direction = eigenV(:,positions);
%   Convert cartesian to spherical coordinates
    
    [phi, theta] = cart2sph(direction(1),direction(2),direction(3));
    x = [S0,D,f,theta,phi];
    
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