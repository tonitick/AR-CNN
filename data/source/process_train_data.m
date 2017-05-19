function process_train_data(setname, sub_img_size, stride, quality)
    data_dir = fullfile('..', 'ProcessedData', 'train');
    if ~exist(data_dir, 'dir')
        mkdir(data_dir);
    end
    
    bin_path = fullfile(data_dir, [setname, '_', num2str(quality), '.bin']);
    bin_file = fopen(bin_path, 'wb');

    file_list = dir(fullfile('..', 'BSDS500', 'data', 'images', setname, '*.jpg'));
    for i = 1:length(file_list)
        img = imread(fullfile('..', 'BSDS500', 'data', 'images', setname, file_list(i).name));
        imwrite(img, 'temp.jpg', 'Quality', quality);
        img_compressed = imread('temp.jpg');
        delete('temp.jpg');
        
        img_size = size(img);
        img_row = img_size(1);
        img_col = img_size(2);
        
        % convert to luminance channel
        for x = 1:img_row
            for y = 1:img_col
                img(x, y, 1) = 0.299 * img(x, y, 1) + 0.587 * img(x, y, 2) + 0.114 * img(x, y, 3);
                img_compressed(x, y, 1) = 0.299 * img_compressed(x, y, 1) + 0.587 * img_compressed(x, y, 2) + 0.114 * img_compressed(x, y, 3);
            end
        end
        
        row_num = floor((img_row - sub_img_size) / stride) + 1;
        col_num = floor((img_col - sub_img_size) / stride) + 1;
        row_shift = floor((img_row - ((row_num - 1) * stride + sub_img_size)) / 2);
        col_shift = floor((img_col - ((col_num - 1) * stride + sub_img_size)) / 2);
        
        for x = 1:row_num
            x_start = row_shift + (x - 1) * stride + 1;
            x_end = row_shift + (x - 1) * stride + sub_img_size;
            for y = 1:col_num
                y_start = col_shift + (y - 1) * stride + 1;
                y_end = col_shift + (y - 1) * stride + sub_img_size;
                
                sub_img = img(x_start:x_end, y_start:y_end, 1);
                sub_img_compressed = img_compressed(x_start:x_end, y_start:y_end, 1);
                fwrite(bin_file, reshape(sub_img, [1, sub_img_size * sub_img_size]), 'uchar');
                fwrite(bin_file, reshape(sub_img_compressed, [1, sub_img_size * sub_img_size]), 'uchar');
                
            end
        end 
    end
    
    fclose(bin_file);
end

