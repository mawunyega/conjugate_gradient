function [A b x] = generateSPDmatrix(n)
% Generate a dense n x n symmetric, positive definite matrix

A = rand(n,n); % generate a random n x n matrix

b = rand(n,1); % generate a random n x 1 vector

x = zeros(n,1); % generate a random n x 1 vector

% construct a symmetric matrix using either
A = 0.5*(A+A'); %OR
%A = A*A';
% The first is significantly faster: O(n^2) compared to O(n^3)

% since A(i,j) < 1 by construction and a symmetric diagonally dominant matrix
%   is symmetric positive definite, which can be ensured by adding nI
A = A + n*eye(n);

Astop = n*n;

bstop = n*1;

fileID = fopen('matrix2048X2048','w');
fileID1 = fopen('vector2048X1','w');
fileID2 = fopen('X2048X1','w');


for i = 1:Astop
    fprintf(fileID,'%.4f\n',A(i));
end

fprintf('beginning vector b');

for i = 1:bstop
    fprintf(fileID1,'%.4f\n',b(i));
end


fprintf('beginning vector x');

for i = 1:bstop
    fprintf(fileID2,'%.1f\n',x(i));
end

end