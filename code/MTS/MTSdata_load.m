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

%% ��ѵ������ÿ���������б���������  %% ��ÿһ��ά�Ƚ�����ռ��ع�
for s_i=1:length(train_data)
    train_data_raw = train_data{s_i}; % ����һ������,12(sensorά��)*20�����ȣ�
    train_label_temp = train_label(s_i);
    disp(train_label_temp);
    
    [h,w] = size(train_data_raw);
    disp([h, w]);
    one_Data = [];
    for i=1:h  % ����ÿһ��ά�Ȳ��ع�
        data_temp=train_data_raw(i,:);
        N=length(data_temp);
        tau=1;
        m=ceil(N/2);
        L=N-(m-1)*tau;%��ռ��е�ĸ���L
        disp(class(data_temp));
        disp(data_temp(1));
        for j=1:L
            for k=1:m
                tmi(j,k)=data_temp((k-1)*tau+j);
                disp(k);
                %X(i,j)=data((i-1)*tau+j);%ԭ�������Ļ�����L��*M�У�Ϊ�˺��淽�㻭�ع���ͼ�񣬻���M*3��ת��
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


%% ����   %% �� ÿһ��ά�Ƚ�����ռ��ع�
for s_i=1:length(test_data)
    test_data_raw = test_data{s_i}; % ����һ������,12(sensorά��)*20�����ȣ�
    test_label_temp = test_label(s_i);
    [h,w] = size(test_data_raw);
    disp([h, w]);
    one_Data_test = [];
    for i=1:h  % ����ÿһ��ά�Ȳ��ع�
        data_temp_test=test_data_raw(i,:);
        N=length(data_temp_test);
        tau=1;
        m=ceil(N/2);
        L=N-(m-1)*tau;%��ռ��е�ĸ���L
        disp(class(data_temp_test));
        disp(data_temp_test(1));
        for j=1:L
            for k=1:m
                tmi(j,k)=data_temp_test((k-1)*tau+j);
                %X(i,j)=data((i-1)*tau+j);%ԭ�������Ļ�����3��*M�У�Ϊ�˺��淽�㻭�ع���ͼ�񣬻���M*3��ת��
            end
        end
        disp(size(tmi));
        one_Data_test = [one_Data_test; tmi];
        
    end
    xlswrite(['MTS/Libras/test/', int2str(s_i),'_',int2str(test_label_temp) , '.xlsx'], one_Data_test);
    clear tmi;
    clear one_Data_test;
    
end



