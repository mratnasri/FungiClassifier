from PIL import ImageFont
import visualkeras
import keras
from keras.models import load_model
model = load_model('../Models/fungi_classifier_model5_DB3_DA_3.h5')
model.summary()
# keras.utils.plot_model(
#    model, "../../Outputs/fungi_classifier_model5_DB3_DA_3_architecture.png", show_shapes=True, rankdir="TB")
model.add(visualkeras.SpacingDummyLayer(spacing=100))
font = ImageFont.truetype("arial.ttf", 30)
'''visualkeras.layered_view(
    model, spacing=0, legend=True, font=font, to_file="../../Outputs/fungi_classifier_model5_DB3_DA_3_architecture_visualkeras_legend_spacing_2.png")'''
visualkeras.layered_view(
    model, scale_z=0.3, scale_xy=2, legend=True, font=font, to_file="../../Outputs/fungi_classifier_model5_DA_3_architecture_visualkeras_legend_scale_3.png")
