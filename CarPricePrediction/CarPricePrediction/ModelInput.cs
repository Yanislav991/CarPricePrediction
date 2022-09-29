namespace CarPricePrediction
{
    public class ModelInput
    {
        public string? Brand { get; set; }
        public string? Model { get; set; }
        public string? TransmissionType { get; set; }
        public string? EngineType { get; set; }
        public string? Type { get; set; }
        public float HorsePower { get; set; }
        public float Year { get; set; }
        public float Price { get; set; }
        public string? Color { get; set; }
    }
}
