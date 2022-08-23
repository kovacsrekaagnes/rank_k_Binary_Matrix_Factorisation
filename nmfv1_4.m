
clear

data_folder_path = 'data_folder'

names = {'tumor_w_missing','hepatitis_w_missing','audio_w_missing','votes_w_missing'};
ks = [2, 5, 10];
results = zeros(size(names,2),3*size(ks,2));

for i = 1:length(names)
    
    name = names{i};
    X = load([data_folder_path, '/', name, '.txt']);
    
    for j = 1:3
        
        k = ks(j);
        % a big initial value of error
        results(i, j) = prod(size(X))
        
        for r = 0:10
            rng(r)
            % Directly call the weighted NMF algorithm to obtain the factor matrices.
            [A_nmf, B_nmf] = wnmfrule(X,k);

            for threshold = 0.1:0.1:0.9
                [A, B] = NMF_to_BMF(A_nmf, B_nmf, threshold);

                Z = A*B >= 1;

                error = sum(sum(abs(X - Z)==1));
                if error < results(i, j)
                    results(i, j) = error;
                    results(i, j+size(ks,2)) = threshold;
                    results(i, j+2*size(ks,2)) = r;
                end
            end
        end
    end
end
        

csvwrite('nmf_completion.csv',results)

function [A, B] = NMF_to_BMF(A_nmf, B_nmf, threshold)
if nargin <=2
    threshold = 0.5;
end
kk = size(A_nmf,2);
for p=1:kk
    maxAp = max(A_nmf(:,p));
    maxBp = max(B_nmf(p,:));
    A_nmf(:,p) = A_nmf(:,p)/ sqrt(maxAp) * sqrt(maxBp);
    B_nmf(:,p) = B_nmf(:,p)/ sqrt(maxBp) * sqrt(maxAp);
end
A = (A_nmf > threshold);
B = (B_nmf > threshold);
    
end