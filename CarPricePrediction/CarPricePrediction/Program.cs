using CarPricePrediction;
using Microsoft.ML;
using Microsoft.ML.AutoML;
using Newtonsoft.Json;


if (!File.Exists("../../../model.zip"))
{
    TrainModel();
}

var context = new MLContext();
var model = context.Model.Load("../../../model.zip", out _);
var predictionEngine = context.Model.CreatePredictionEngine<ModelInput, ModelOutput>(model);
//"Brand":"honda","Model":"Cr-v","Price":13900.0,"Type":"Джип","EngineType":"Дизел","TransmissionType":"Ръчна","ManufactureDate":"2008-01-01T00:00:00","HorsePower":140,"Color":"Тъмно син мет."}

var testModel = new ModelInput()
{
    Brand = "audi",
    Color = "Черен",
    EngineType = "Дизел",
    HorsePower = 190,
    Model = "А4",
    TransmissionType = "Автоматична",
    Type = "Седан",
    Year = 2020
}; 
var prediction = predictionEngine.Predict(testModel);
Console.WriteLine(prediction.Score);
Console.ReadLine();


static void TrainModel()
{
    var data = GetData().Where(y => y.ManufactureDate != null && y.Brand != null && y.HorsePower != null)
        .Select(x => new ModelInput
        {
            Brand = x.Brand,
            Color = x.Color,
            EngineType = x.EngineType,
            HorsePower = (float)x.HorsePower,
            Model = x.Model,
            Price = x.Price,
            TransmissionType = x.TransmissionType,
            Type = x.Type,
            Year = x.ManufactureDate.Value.Year
        }).ToList();

    var context = new MLContext();
    var dataView = context.Data.LoadFromEnumerable<ModelInput>(data);
    var pipeline = context.Transforms.Categorical.OneHotEncoding(new InputOutputColumnPair[]
    {
    new InputOutputColumnPair("Brand", "Brand"),
    new InputOutputColumnPair("Model", "Model"),
    new InputOutputColumnPair("TransmissionType", "TransmissionType"),
    new InputOutputColumnPair("EngineType", "EngineType"),
    new InputOutputColumnPair("Type", "Type"),
    new InputOutputColumnPair("Color", "Color"),
    }).Append(context.Transforms.Concatenate(outputColumnName: "Features",
                                            nameof(ModelInput.Year),
                                            nameof(ModelInput.HorsePower),
                                            nameof(ModelInput.Brand),
                                            nameof(ModelInput.Model),
                                            nameof(ModelInput.TransmissionType),
                                            nameof(ModelInput.EngineType),
                                            nameof(ModelInput.Type),
                                            nameof(ModelInput.Color)));



    var trainer = context.Regression.Trainers.LightGbm(labelColumnName: "Price",
                                                                     featureColumnName: "Features");

    var trainingPipeLine = pipeline.Append(trainer);
    var model = trainingPipeLine.Fit(dataView);

    context.Model.Save(model, dataView.Schema, "../../../model.zip");
}

static List<CarModel> GetData()
{
    var fileText = File.ReadAllText("../../../cars.json");
    return JsonConvert.DeserializeObject<List<CarModel>>(fileText, new JsonSerializerSettings() { NullValueHandling = NullValueHandling.Ignore });
}