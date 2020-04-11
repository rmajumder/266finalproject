def get_scores(path, start_line_text_idx, start_text_score_idx, start_score_idx, start_iter_indx, end_iter_indx):

    results = []
    max_results = []
    
    for i in range(start_iter_indx, end_iter_indx):
        iter_score = []
        file1 = open(path + str(i) + '.txt', 'r') 
        lines = file1.readlines() 
        for line in lines:
            if line.startswith(start_line_text_idx):
                scores = line.split(',')

                for sc in scores:
                    if sc.strip().startswith(start_text_score_idx):
                        sc_fl = float(sc[start_score_idx:])
                        iter_score.append(sc_fl)

        results.append(iter_score)
        max_results.append(max(iter_score))
        
        file1.close()

    return results, max_results

def get_performance_data(path, score_type, file_start_idx, file_end_idx):
    if score_type == 'dev_acc':
        return get_scores(path, 'train loss', 'dev acc', 9, file_start_idx, file_end_idx)
    elif score_type == 'loss':
        return get_scores(path, 'train loss', 'train loss', 12, file_start_idx, file_end_idx)
    elif score_type == 'test_acc':
        return get_scores(path, 'test acc', 'test acc', 10, file_start_idx, file_end_idx)
    elif score_type == 'test_prec':
        return get_scores(path, 'test acc', 'test precision', 16, file_start_idx, file_end_idx)
    elif score_type == 'test_recall':
        return get_scores(path, 'test acc', 'test recall', 13, file_start_idx, file_end_idx)
    elif score_type == 'test_f1':
        return get_scores(path, 'test acc', 'test f1', 9, file_start_idx, file_end_idx)
    
    return []

def get_raw_200(score_type):
    path = 'data/200/raw/results-200-raw-15iter-'
    return get_performance_data(path, score_type, 1, 12)

def get_raw_200_max(score_type):
    path = 'data/200/raw/results-200-raw-15iter-'
    return get_performance_data(path, score_type, 10, 11)

def get_raw_200_xavier(score_type):
    path = 'data/200/raw-xavier/results-200-raw-xavier-init-'
    return get_performance_data(path, score_type, 1, 11)

def get_raw_200_xavier_max(score_type):
    path = 'data/200/raw-xavier/results-200-raw-xavier-init-'
    return get_performance_data(path, score_type, 10, 11)

def get_raw_200_clean(score_type):
    path = 'data/200/clean/results-200-clean-'
    return get_performance_data(path, score_type, 1, 11)

def get_raw_200_clean_max(score_type):
    path = 'data/200/clean/results-200-clean-'
    return get_performance_data(path, score_type, 10, 11)

def get_raw_400(score_type):
    path = 'data/400/results-400-raw-part'
    return get_performance_data(path, score_type, 1, 11)

def get_raw_400_max(score_type):
    path = 'data/400/results-400-raw-part'
    return get_performance_data(path, score_type, 8, 9)

def get_raw_100(score_type):
    path = 'data/100/results-100-raw-part'
    return get_performance_data(path, score_type, 1, 11)

def get_raw_100_max(score_type):
    path = 'data/100/results-100-raw-part'
    return get_performance_data(path, score_type, 2, 3)
