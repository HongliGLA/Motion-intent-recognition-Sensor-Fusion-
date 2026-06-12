%% ============================================================
% convert_DB6_S2_to_csv.m
%
% DB6 Subject 2 .mat -> 200Hz merged CSV (High-Speed & High-Precision)
%
% Input:
%   S2_D1_T1.mat
%   S2_D1_T2.mat
%   ...
%   S2_D5_T2.mat
%
% Output folder:
%   S2_200Hz_merged_csv_high_precision
%
% Output CSV columns:
%   1-16   : EMG
%   17-64  : ACC
%   65     : restimulus
%   66     : rerepetition
%
% Python indexing:
%   emg    = data[:, 0:16]
%   acc    = data[:, 16:64]
%   label  = data[:, 64]
%   rep    = data[:, 65]
%% ============================================================
clear;
clc;

%% ============================================================
% 1. Path setting
% ============================================================
input_folder = pwd;
% Output folder
output_folder = fullfile(pwd, "S3_200Hz_merged_csv_high_precision");
if ~exist(output_folder, "dir")
    mkdir(output_folder);
end

fprintf("Current MATLAB folder:\n%s\n", pwd);
fprintf("Input folder:\n%s\n", input_folder);
fprintf("Output folder:\n%s\n", output_folder);

%% ============================================================
% 2. Sampling rate setting
% ============================================================
original_fs = 2000;
target_fs = 200;
downsample_factor = original_fs / target_fs;
if mod(downsample_factor, 1) ~= 0
    error("Downsample factor is not an integer.");
end
downsample_factor = round(downsample_factor);

fprintf("\nOriginal sampling rate: %d Hz\n", original_fs);
fprintf("Target sampling rate: %d Hz\n", target_fs);
fprintf("Downsample factor: %d\n", downsample_factor);

%% ============================================================
% 3. Find all S2 .mat files 
% ============================================================
mat_files = dir(fullfile(input_folder, "S3_D*_T*.mat"));
if isempty(mat_files)
    error("No S3_D*_T*.mat files found in current folder.");
end

fprintf("\nFound %d files.\n", length(mat_files));
for k = 1:length(mat_files)
    fprintf("  %s\n", mat_files(k).name);
end

%% ============================================================
% 4. Process each .mat file
% ============================================================
for k = 1:length(mat_files)
    mat_name = mat_files(k).name;
    mat_path = fullfile(input_folder, mat_name);
    
    fprintf("\n============================================================\n");
    fprintf("Processing %d/%d: %s\n", k, length(mat_files), mat_name);
    
    %% --------------------------------------------------------
    % Load .mat file
    % --------------------------------------------------------
    s = load(mat_path);
    
    %% --------------------------------------------------------
    % Check required variables
    % --------------------------------------------------------
    required_vars = ["emg", "acc", "restimulus", "rerepetition"];
    for v = 1:length(required_vars)
        var_name = required_vars(v);
        if ~isfield(s, var_name)
            error("File %s does not contain variable: %s", mat_name, var_name);
        end
    end
    
    %% --------------------------------------------------------
    % Extract variables
    % --------------------------------------------------------
    emg = double(s.emg);
    acc = double(s.acc);
    restimulus = double(s.restimulus(:));
    rerepetition = double(s.rerepetition(:));
    
    %% --------------------------------------------------------
    % Print original information
    % --------------------------------------------------------
    fprintf("Raw EMG shape: %d x %d\n", size(emg, 1), size(emg, 2));
    fprintf("Raw ACC shape: %d x %d\n", size(acc, 1), size(acc, 2));
    fprintf("Raw restimulus length: %d\n", length(restimulus));
    fprintf("Raw rerepetition length: %d\n", length(rerepetition));
    
    if size(emg, 2) ~= 16
        warning("EMG channel number is not 16. Current: %d", size(emg, 2));
    end
    if size(acc, 2) ~= 48
        warning("ACC channel number is not 48. Current: %d", size(acc, 2));
    end
    
    %% --------------------------------------------------------
    % Align length
    % --------------------------------------------------------
    min_len = min([
        size(emg, 1), ...
        size(acc, 1), ...
        length(restimulus), ...
        length(rerepetition)
    ]);
    emg = emg(1:min_len, :);
    acc = acc(1:min_len, :);
    restimulus = restimulus(1:min_len);
    rerepetition = rerepetition(1:min_len);
    fprintf("Aligned length: %d\n", min_len);
    
    %% --------------------------------------------------------
    % Downsample from 2000 Hz to 200 Hz
    % --------------------------------------------------------
    idx = 1:downsample_factor:min_len;
    emg_200 = emg(idx, :);
    acc_200 = acc(idx, :);
    restimulus_200 = restimulus(idx);
    rerepetition_200 = rerepetition(idx);
    
    %% --------------------------------------------------------
    % Merge into one matrix
    % --------------------------------------------------------
    merged_data = [
        emg_200, ...
        acc_200, ...
        restimulus_200, ...
        rerepetition_200
    ];
    fprintf("Merged shape: %d x %d\n", size(merged_data, 1), size(merged_data, 2));
    if size(merged_data, 2) ~= 66
        warning("Merged data columns are not 66. Current: %d", size(merged_data, 2));
    end
    
    %% --------------------------------------------------------
    % 终极修复版：高速保存高精度 CSV (Vectorized fprintf)
    % --------------------------------------------------------
    [~, base_name, ~] = fileparts(mat_name);
    output_name = base_name + "_200Hz_merged.csv";
    output_path = fullfile(output_folder, output_name);
    
    fid = fopen(output_path, "w");
    if fid == -1
        error("Cannot open output file: %s", output_path);
    end
    
    % 64 个高精度信号列 + 2 个整数标签列
    format_line = [repmat('%.12g,', 1, 64), '%.0f,%.0f\n'];
    
    % 【核心加速】：去掉 for 循环！直接将矩阵转置后一次性写入！
    fprintf(fid, format_line, merged_data');
    
    fclose(fid);
    fprintf("Saved high-precision CSV:\n%s\n", output_path);
    
    %% --------------------------------------------------------
    % 高效检查数据 (在内存中检查，不重新读取大文件)
    % --------------------------------------------------------
    check_emg = merged_data(:, 1:16);
    check_acc = merged_data(:, 17:64);
    check_label = merged_data(:, 65);
    check_rep = merged_data(:, 66);
    
    fprintf("CHECK (Memory) EMG min = %.12f, max = %.12f, nonzero = %d\n", ...
        min(check_emg(:)), max(check_emg(:)), nnz(check_emg));
    fprintf("CHECK (Memory) ACC min = %.12f, max = %.12f, nonzero = %d\n", ...
        min(check_acc(:)), max(check_acc(:)), nnz(check_acc));
    fprintf("CHECK unique restimulus labels:\n");
    disp(unique(check_label)'-0); 
    fprintf("CHECK unique rerepetition labels:\n");
    disp(unique(check_rep)'-0);
end

fprintf("\n============================================================\n");
fprintf("All S2 files converted successfully with high speed and high precision.\n");
fprintf("Output folder:\n%s\n", output_folder);
fprintf("============================================================\n");