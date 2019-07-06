clc;clear all; close all; fclose all;
addpath(fullfile('..','JPEG_Toolbox'));

% dataset_paths = [
%     "~/myworks/stego/datasets/BSDS300/images",
%     "~/myworks/stego/datasets/BSDS500/BSDS500/data/images",
%     "~/myworks/stego/datasets/INRIA-jpg1",
%     "~/myworks/stego/datasets/INRIA-jpg2"
% ];
payloads = [0.05, 0.1, 0.2, 0.4];

path = '/home/emadhelmi/myworks/stego/datasets/INRIA-jpg1';
addpath(fullfile(path, 'total'));
for p=1:length(payloads)
    addpath(fullfile(path, 'stego', num2str(payloads(p))))
    fprintf('Reading files from %s\n', path);
    files = dir(fullfile(path, 'total', '*.jpg'));
    fprintf('\tTotal <.jpg>s in this path %d files\n', length(files));
    embed(fullfile(path), payloads(p))
end
