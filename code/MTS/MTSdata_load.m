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

%% 对训练集的每个样本进行遍历并操作  %% 对每一个维度进行相空间重构
for s_i=1:length(train_data)
    train_data_raw = train_data{s_i}; % 这是一个样本,12(sensor维度)*20（长度）
    train_label_temp = train_label(s_i);
    disp(train_label_temp);
    
    [h,w] = size(train_data_raw);
    disp([h, w]);
    one_Data = [];
    for i=1:h  % 遍历每一个维度并重构
        data_temp=train_data_raw(i,:);
        N=length(data_temp);
        tau=1;
        m=ceil(N/2);
        L=N-(m-1)*tau;%相空间中点的个数L
        disp(class(data_temp));
        disp(data_temp(1));
        for j=1:L
            for k=1:m
                tmi(j,k)=data_temp((k-1)*tau+j);
                disp(k);
                %X(i,j)=data((i-1)*tau+j);%原本这样的话就是L行*M列，为了后面方便画重构的图像，换成M*3好转换
            end
        end
        disp(tmi);
        disp(class(tmi));
        one_Data = [one_Data; tmi];
        
    end
    xlswrite(['MTS/Libras/train/', int2str(s_i),'_',int2str(train_label_temp), '.xlsx'], one_Data);
    clear tmi;
    clear one_Data;
    

end


%% 测试   %% 对 每一个维度进行相空间重构
for s_i=1:length(test_data)
    test_data_raw = test_data{s_i}; % 这是一个样本,12(sensor维度)*20（长度）
    test_label_temp = test_label(s_i);
    [h,w] = size(test_data_raw);
    disp([h, w]);
    one_Data_test = [];
    for i=1:h  % 遍历每一个维度并重构
        data_temp_test=test_data_raw(i,:);
        N=length(data_temp_test);
        tau=1;
        m=ceil(N/2);
        L=N-(m-1)*tau;%相空间中点的个数L
        disp(class(data_temp_test));
        disp(data_temp_test(1));
        for j=1:L
            for k=1:m
                tmi(j,k)=data_temp_test((k-1)*tau+j);
                %X(i,j)=data((i-1)*tau+j);%原本这样的话就是3行*M列，为了后面方便画重构的图像，换成M*3好转换
            end
        end
        disp(size(tmi));
        one_Data_test = [one_Data_test; tmi];
        
    end
    xlswrite(['MTS/Libras/test/', int2str(s_i),'_',int2str(test_label_temp) , '.xlsx'], one_Data_test);
    clear tmi;
    clear one_Data_test;
    
end



