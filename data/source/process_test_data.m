function process_test_data(quality)
    data_dir = fullfile('..', 'ProcessedData', 'test');
    if ~exist(data_dir, 'dir')
        mkdir(data_dir);
    end
    
    file_list = dir(fullfile('..', 'databaserelease2', 'refimgs', '*.bmp'));
    for i = 1:length(file_list)
        filename = file_list(i).name(1:(length(file_list(i).name) - 4));
        bin_path = fullfile(data_dir, [filename, '_', num2str(quality), '.bin']);
        bin_file = fopen(bin_path, 'wb');
        
        img = imread(fullfile('..', 'databaserelease2', 'refimgs', file_list(i).name));
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
                img_compressed(x, y, 1) = 0.299 * img_compressed(x, y, 1) + 0.587 * img_compressed(x, y, 2) + 0.114 * img(x, y, 3);
            end
        end
        
        fwrite(bin_file, img_row, 'int');
        fwrite(bin_file, img_col, 'int');
        fwrite(bin_file, reshape(img(:, :, 1), [1, img_row * img_col]), 'uchar');
        fwrite(bin_file, reshape(img_compressed(:, :, 1), [1, img_row * img_col]), 'uchar');

        fclose(bin_file);
    end
end

