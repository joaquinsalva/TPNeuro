#%%
from main import * 

#%%
nuevo_volume_columns = ['group'] + ['volume_' + nombre for nombre in volume_columns]
nuevo_thickness_columns = ['group'] + ['thickness_' + nombre for nombre in thickness_columns]

thickness_all = [bip_sch_thickness, bip_cnt_thickness, sch_cnt_thickness, thickness2]
volume_all = [bip_sch_volume2, bip_cnt_volume2, sch_cnt_volume2, volume2]

for i in range(len(thickness_all)):
    thickness_all[i].columns=nuevo_thickness_columns

for i in range(len(volume_all)):
    volume_all[i].columns=nuevo_volume_columns

data = []
for i in range(4):
    df_concat = pd.concat([thickness_all[i], volume_all[i].drop('group', axis=1)], axis=1)
    data.append(df_concat)


#%%

X_types = {
    # “gm”: [“gm_.*”],
    # “wm”: [“wm_.*”],
    "thickness": ["thickness_.*"],
    "volume": ["volume_.*"],
}

X_names = X_types["thickness"] + X_types["volume"]

model_thickness = PipelineCreator(problem_type="classification",  apply_to="thickness")
model_thickness.add("filter_columns", apply_to="*", keep="thickness")

model_volume = PipelineCreator(problem_type="classification", apply_to="volume")
model_volume.add("filter_columns", apply_to="*", keep="volume")

model_gm = PipelineCreator(problem_type="classification", apply_to="gm")
model_gm.add("filter_columns", apply_to="*", keep="gm")

model_wm = PipelineCreator(problem_type="classification", apply_to="wm")
model_wm.add("filter_columns", apply_to="*", keep="wm")


model = PipelineCreator(problem_type="classification")
model.add(
    "stacking",
    estimators=[[("model_thickness", model_thickness),("model_volume", model_volume),
                 ("model_gm", model_gm), ("model_wm", model_wm)]],
    apply_to="*",
)


#%%
res = []
for i in range(len(data)):
    scores, final = run_cross_validation(
        X=X_names,
        X_types=X_types,
        y="target",
        data=data[i],
        model=model,
        seed=200,
        return_estimator="final",
    )
    best_params = final.best_estimator_.get_params()
    res.append((scores, best_params))
