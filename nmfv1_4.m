
clear

data_folder_path = 'data_folder'

names = {'tumor_w_missing','hepatitis_w_missing','audio_w_missing','votes_w_missing'};
results = [];
ks = [2, 5, 10];

for i = 1:length(names)
    
    name = names{i};
    X = load([data_folder_path, '/', name, '.txt']);
    
    for j = 1:3
        
        k = ks(j);
        
        % Directly call the weighted NMF algorithm to obtain the factor matrices.
        [A_nmf, B_nmf] = wnmfrule(X,k);

        A = A_nmf > 0.5;
        B = B_nmf > 0.5;

        Z = A*B >= 1;

        results(i, j) = sum(sum(abs(X - Z)==1));
    end
        
end
        

csvwrite('nmf_completion.csv',results)

