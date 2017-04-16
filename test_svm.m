clc;
clear all;
close all;

file = 'letterj.csv'; % Dataset
K = 5;
% Reading training file
for j = 1:4
    for i=1:K
        train_data = dlmread(strcat(file,'train',int2str(j),int2str(i),'.csv'));
        test_data = dlmread(strcat(file,'test',int2str(j),int2str(i),'.csv'));
        test_op = test_data(:,end);
        m = size(train_data,1);
        prob_doc = dlmread(strcat(file,'train',int2str(j),int2str(i),'_report.txt'));
        prob_doc = prob_doc(prob_doc(:,2)==-1,end);
        prob_doc = prob_doc ./ sum(prob_doc); % normalization
        % TODUS
        [prediction,score] = TODUS(train_data,test_data,'tree',prob_doc);
        % Storing the Precision and Recall Values
		output(((j-1)*K)+i) = confusionmatStats(test_op,prediction);
		[~,~,~,auc(((j-1)*K)+i,1)] = perfcurve(test_op,score(:,2),'1');
    end
end