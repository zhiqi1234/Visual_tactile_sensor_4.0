% 假设你的标定结果变量名为 stereoParams
% 如果你的变量名不同，请在下方修改，例如： params = result_variable;
if ~exist('stereoParams', 'var')
    error('错误: 工作区中未找到名为 stereoParams 的变量。请先加载标定结果。');
end

% 定义输出文件夹
outputDir = 'calibration';
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
    fprintf('已创建文件夹: %s\n', outputDir);
end

%% 1. 处理内参矩阵 (Intrinsics)
% MATLAB store format: [fx 0 0; s fy 0; cx cy 1]' (Transpose of standard)
% OpenCV format: [fx s cx; 0 fy cy; 0 0 1]
% 操作: 需要对 MATLAB 的 IntrinsicMatrix 进行转置 (')

K1 = stereoParams.CameraParameters1.IntrinsicMatrix';
K2 = stereoParams.CameraParameters2.IntrinsicMatrix';

%% 2. 处理畸变系数 (Distortion Coefficients)
% MATLAB: RadialDistortion [k1, k2, (k3)], TangentialDistortion [p1, p2]
% OpenCV format: [k1, k2, p1, p2, k3]

function D = convertDistortion(camParams)
    r = camParams.RadialDistortion;
    t = camParams.TangentialDistortion;
    
    % 检查是否有 k3 (3个径向参数)
    k3 = 0;
    if numel(r) >= 3
        k3 = r(3);
    end
    
    % OpenCV 顺序: k1, k2, p1, p2, k3
    D = [r(1), r(2), t(1), t(2), k3];
end

D1 = convertDistortion(stereoParams.CameraParameters1);
D2 = convertDistortion(stereoParams.CameraParameters2);

%% 3. 处理旋转和平移 (Rotation & Translation)
% MATLAB Rotation: R_matlab. point_new = point_old * R_matlab
% OpenCV Rotation: R_opencv. point_new = R_opencv * point_old
% 操作: 需要转置 R

R = stereoParams.RotationOfCamera2'; 
T = stereoParams.TranslationOfCamera2'; % 转置为列向量 (3x1)

%% 4. 写入 TXT 文件
% 使用高精度 (%.12f) 避免精度丢失

writeData(fullfile(outputDir, 'K1.txt'), K1);
writeData(fullfile(outputDir, 'K2.txt'), K2);
writeData(fullfile(outputDir, 'D1.txt'), D1);
writeData(fullfile(outputDir, 'D2.txt'), D2);
writeData(fullfile(outputDir, 'R.txt'), R);
writeData(fullfile(outputDir, 'T.txt'), T);

fprintf('所有标定文件已成功导出至 "%s" 文件夹。\n', outputDir);

%% 辅助写入函数
function writeData(filename, matrix)
    fileID = fopen(filename, 'w');
    if fileID == -1
        error('无法打开文件写入: %s', filename);
    end
    
    [rows, cols] = size(matrix);
    for i = 1:rows
        for j = 1:cols
            fprintf(fileID, '%.12e ', matrix(i, j));
        end
        fprintf(fileID, '\n'); % 换行
    end
    fclose(fileID);
end