data=load('mtsdata(review_13)\Libras/Libras.mat');
disp(data);
disp(class(data));
disp(size(data));
c_data = struct2cell(data);
disp(class(c_data));
disp(size(c_data));
X = c_data{1};
disp(class(X));
train_data = X.train;% ѵ������
test_data = X.test; %��������
train_label = X.trainlabels;
test_label = X.testlabels;
disp(class(test_data));
% train_data��ȫ������������ȡ������
disp(train_data)
disp(size(train_data))
%% ��ѵ������ÿ���������б���������  %% ��ÿһ��ά�Ƚ�����ռ��ع�
for train_i=1:length(train_data)
    train_data_raw = train_data{train_i}; % ����һ������,12(sensorά��)*20�����ȣ�
    train_label_temp = train_label(train_i);
    disp(train_label_temp);
    [h,w] = size(train_data_raw);
    disp([h, w]);
    xlswrite(['raw_MTS/Libras/train/', int2str(train_i),'_',int2str(train_label_temp), '.xlsx'], train_data_raw);
    clear train_data_raw;
end

% 
%% ����   %% �� ÿһ��ά�Ƚ�����ռ��ع�
for test_i=1:length(test_data)
    test_data_raw = test_data{test_i}; % ����һ������,12(sensorά��)*20�����ȣ�
    test_label_temp = test_label(test_i);
    [h,w] = size(test_data_raw);
    disp([h, w]);
    xlswrite(['raw_MTS/Libras/test/', int2str(test_i),'_',int2str(test_label_temp) , '.xlsx'], test_data_raw);
    
    clear test_data_raw;
    
end
