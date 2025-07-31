import joblib
import pandas as pd

def load_model(model_path='scripts/random_forest_model.joblib'):
    model = joblib.load(model_path)
    print("✅ Model loaded successfully!")
    return model

def load_scaler(scaler_path='scripts/scaler.joblib'):
    scaler = joblib.load(scaler_path)
    print("✅ Scaler loaded successfully!")
    return scaler

def predict(model, scaler, input_df):
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)
    return prediction

if __name__ == "__main__":
    model = load_model()
    scaler = load_scaler()

    sample_input = pd.DataFrame([[
        0,  # Time
        -1.3598071336738, -0.0727811733098497, 2.53634673796914,
        1.37815522427443, -0.338320769942518, 0.462387777762292,
        0.239598554061257, 0.0986979012610507, 0.363786969611213,
        0.0907941719789316, -0.551599533260813, -0.617800855762348,
        -0.991389847235408, -0.311169353699879, 1.46817697209427,
        -0.470400525259478, 0.207971241929242, 0.0257905801985591,
        0.403992960255733, 0.251412098239705, -0.018306777944153,
        0.277837575558899, -0.110473910188767, 0.0669280749146731,
        0.128539358273528, -0.189114843888824, 0.133558376740387,
        -0.0210530534538215, 149.62
    ]], columns=['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount'])

    result = predict(model, scaler, sample_input)

    print("🚨 Fraud Detected!" if result[0] == 1 else "✅ Transaction is Normal")
