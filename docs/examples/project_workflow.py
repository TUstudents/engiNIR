import numpy as np

from enginir import PreprocessingRecipe, Project
from enginir.io import save_hdf5

# A synthetic scan of 8 samples, 5 frames each: 40 frames x 10 pixels x 50
# bands of raw counts whose absorption at band 25 grows with the API content.
rng = np.random.default_rng(0)
wavelengths = np.linspace(1000, 1700, 50)
content = np.linspace(1, 9, 8)
peak = np.exp(-((np.arange(50) - 25) / 6) ** 2)
cube = (2500 - 100 * np.repeat(content, 5)[:, None, None] * peak
        + rng.normal(0, 5, (40, 10, 50))).astype(np.uint16)
save_hdf5("scan.h5", cube, wavelengths)

project = Project.create("tablets.h5")
scan = project.add_dataset("scan.h5")                 # calibrated to absorbance
project.add_target("API content", unit="% w/w")
for i, value in enumerate(content):
    scan.add_roi(f"tablet {i + 1}", frames=(5 * i, 5 * i + 4), rows=(0, 9),
                 values={"API content": value})

model = project.fit("API content", recipe=PreprocessingRecipe(snv=True), n_components=2)
print(model)

prediction = model.predict("scan.h5")
print(prediction.map.reshape(8, 5, 10).mean(axis=(1, 2)).round(1))   # mean per tablet
