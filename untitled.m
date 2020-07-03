% dataset_name = "BOWS2OrigEp3";
dataset_name = "BOSSbase_1.01";
source_dir = "../datasets/" + dataset_name + "/total/original";
imresize_dir = "../datasets/" + dataset_name + "/total/imresize";
imcrop_dir = "../datasets/" + dataset_name + "/total/imcrop";
subsample_dir = "../datasets/" + dataset_name + "/total/subsample";

for file = dir(source_dir + "/*.pgm")'
    % Execute the new images
    original_image = imread(source_dir + "/" + file.name);
    imresize_result = imresize(original_image, [256 256]);
    imcrop_result = imcrop(original_image, [128 128 255 255]);
    subsample_result = original_image(1:2:end, 1:2:end);
    
    % Write the new images
    imwrite(imresize_result, imresize_dir + "/" + file.name);
    imwrite(imcrop_result, imcrop_dir + "/" + file.name);
    imwrite(subsample_result, subsample_dir + "/" + file.name);
    
    fprintf(file.name + " " + "finished.\n");
end