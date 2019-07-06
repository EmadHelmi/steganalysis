function embed(dir_path, pld)
    % fprintf("######## START ########\n")
    cover_path = fullfile(dir_path, 'total');
    stego_path = fullfile(dir_path, 'stego', num2str(pld));
    files = dir(fullfile(cover_path, '*.jpg'));
    for i=1:length(files)
        filename = files(i).name;
        fprintf("\t\tEmbedding with payload %.2f into %s", pld, filename)
        s = tic;
        S_STRUCT = J_UNIWARD(fullfile(cover_path, filename), pld);
        jpeg_write(S_STRUCT, fullfile(stego_path, filename));
        e = toc(s);
        fprintf(" finished in %.2fs\n", e);
    end
    % fprintf("########  END  ########\n")
end