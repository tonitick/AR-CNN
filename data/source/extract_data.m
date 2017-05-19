function extract_data()
    % train data
    process_train_data('train', 32, 10, 10);
    process_train_data('train', 32, 10, 20);
    process_train_data('train', 32, 10, 30);
    process_train_data('train', 32, 10, 40);
    
    process_train_data('test', 32, 10, 10);
    process_train_data('test', 32, 10, 20);
    process_train_data('test', 32, 10, 30);
    process_train_data('test', 32, 10, 40);
    
    process_train_data('val', 32, 10, 10);
    process_train_data('val', 32, 10, 20);
    process_train_data('val', 32, 10, 30);
    process_train_data('val', 32, 10, 40);
    
    % test data
    process_test_data(10);
    process_test_data(20);
    process_test_data(30);
    process_test_data(40);
end

