import os
import pandas as pd
from ml_engine.pipeline import PipelineManager

pm = PipelineManager()
session = pm.create_session()

df = pd.DataFrame({
    'x1': [1,2,3,4,5,6,7,8,9,10] * 5,
    'x2': [10,9,8,7,6,5,4,3,2,1] * 5,
    'y': [0,1,0,1,0,1,0,1,0,1] * 5
})
df.to_csv('dummy.csv', index=False)

try:
    pm.upload_and_profile(session.session_id, 'dummy.csv', '')
    session.profile['target_column'] = 'y'
    session.profile['problem_type'] = 'classification'
    session.is_timeseries = False
    
    pm.clean_and_transform(session.session_id)
    import time
    while session.current_step != 'train':
        time.sleep(0.5)
    
    pm.train(session.session_id)
    while session.current_step != 'results':
        time.sleep(0.5)
        if session.status == 'error':
            print('Error in training:', session.progress_message)
            break
    
    print("DIAGNOSTICS:")
    diag = pm.get_diagnostics(session.session_id)
    print(diag)
    
    print("\nWHATIF:")
    wi = pm.run_whatif(session.session_id, 0, 'x1', 5)
    print(wi)
except Exception as e:
    import traceback
    traceback.print_exc()
