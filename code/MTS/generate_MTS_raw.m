data=load('mtsdata(review_13)\Libras/Libras.mat');
disp(data);
disp(class(data));
disp(size(data));
c_data = struct2cell(data);
disp(class(c_data));
disp(size(c_data));
X = c_data{1};
disp(class(X));
train_data = X.train;% 训练样本
test_data = X.test; %测试样本
train_label = X.trainlabels;
test_label = X.testlabels;
disp(class(test_data));
% train_data是全部的样本，提取样本的
disp(train_data)
disp(size(train_data))
%% 对训练集的每个样本进行遍历并操作  %% 对每一个维度进行相空间重构
for train_i=1:length(train_data)
    train_data_raw = train_data{train_i}; % 这是一个样本,12(sensor维度)*20（长度）
    train_label_temp = train_label(train_i);
    disp(train_label_temp);
    [h,w] = size(train_data_raw);
    disp([h, w]);
    xlswrite(['raw_MTS/Libras/train/', int2str(train_i),'_',int2str(train_label_temp), '.xlsx'], train_data_raw);
    clear train_data_raw;
end

% 
%% 测试   %% 对 每一个维度进行相空间重构
for test_i=1:length(test_data)
    test_data_raw = test_data{test_i}; % 这是一个样本,12(sensor维度)*20（长度）
    test_label_temp = test_label(test_i);
    [h,w] = size(test_data_raw);
    disp([h, w]);
    xlswrite(['raw_MTS/Libras/test/', int2str(test_i),'_',int2str(test_label_temp) , '.xlsx'], test_data_raw);
    
    clear test_data_raw;
    
end
