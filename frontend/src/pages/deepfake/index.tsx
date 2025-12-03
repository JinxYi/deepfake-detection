import { Layout } from "@/components/layouts";
import {
  Box,
  Button,
  Card,
  CardContent,
  Divider,
  IconButton,
  Typography,
} from "@mui/joy";
import { PieChart } from "@mui/x-charts/PieChart";
import { Grid2 as Grid } from "@mui/material";
import { UploadButton } from "@/components/upload-button";
import { useState } from "react";
import { niceBytes } from "@/utils/bytes-converter";
import DeleteOutlineIcon from "@mui/icons-material/DeleteOutline";
import { CustomLoader } from "@/components/custom-loader";
import CheckCircleIcon from "@mui/icons-material/CheckCircle";
import { ImageReport } from "@/interface/Results";
import { WarningOutlined } from "@mui/icons-material";
import ArrowBackIcon from '@mui/icons-material/ArrowBack';
export const DeepfakeDectector = () => {
  const [loading, setLoading] = useState(false);
  const [step, setStep] = useState(0);
  const [files, setFiles] = useState<File[]>([]);
  const [imageResults, setImageResults] = useState<ImageReport>();
  const [prediction, setPrediction] = useState<string>();

  const handleScan = async () => {
    setLoading(true);
    setStep(-1);
    const formData = new FormData();
    files.forEach((file) => {
      formData.append(file.type.startsWith("image/") ? "image" : "file", file); // based on file type
    });
    try {
      if (files[0].type.startsWith("image/")) {
        const response = await fetch("http://127.0.0.1:5000/predict", {
          method: "POST",
          body: formData,
        });

        const results = await response.json();
        const prediction = results.prediction_label;

        setPrediction(prediction);
        setImageResults(results);
        setStep(1);
      }
    } catch (error) {
      console.error("Error scanning files:", error);
      setStep(0);
      alert("An unexpected error occured.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <Layout>
        <Box
          style={{
            maxWidth: 650,
            margin: "auto",
            display: "flex",
            flexDirection: "column",
          }}
        >
          {step === 0 && (
            <Grid
              container
              direction="column"
              spacing={2}
              sx={{ marginTop: "20vh" }}
              flex={"1 1 auto"}
              justifyContent={"center"}
              alignItems={"center"}
            >
              <Card sx={{ padding: "1rem" }}>
                <Typography
                  level="title-md"
                  gutterBottom
                  // sx={{ fontWeight: "regular" }}
                >
                  Upload the image which you want to check for
                  deepfakes.
                </Typography>

                <UploadButton files={files} setFiles={setFiles} />
                <Typography level="body-xs" sx={{ color: "text.secondary" }}>
                    *Note that this detector analyses the most prominent face in the image.
                  </Typography>
                {files.map((file, index) => (
                  <Card key={index} sx={{ display: "flex", width: "100%" }}>
                    <Grid container spacing={2} sx={{ flexGrow: 1 }}>
                      <Grid size={6}>
                        {file.type.startsWith("image/") && (
                          <img
                            src={URL.createObjectURL(file)}
                            alt={file.name}
                            style={{ width: "100%", height: "auto" }}
                          />
                        )}
                      </Grid>
                      <Grid size={6}>
                        <Box
                          sx={{
                            display: "flex",
                            justifyContent: "space-between",
                            alignSelf: "flex-start",
                          }}
                        >
                          <CardContent sx={{ flex: "1 0 auto" }}>
                            <Typography level="title-md">
                              {file.name}
                            </Typography>
                            <Typography
                              level="body-sm"
                              component="div"
                              sx={{ color: "text.secondary" }}
                            >
                              {niceBytes(file.size)}
                            </Typography>
                          </CardContent>
                          <IconButton
                            aria-label="delete"
                            onClick={() => {
                              setFiles(files.filter((f) => f !== file));
                            }}
                          >
                            <DeleteOutlineIcon />
                          </IconButton>
                        </Box>
                      </Grid>
                    </Grid>
                  </Card>
                ))}
                <Button
                  variant="solid"
                  loading={loading}
                  disabled={files.length == 0}
                  sx={{ width: "100%" }}
                  onClick={() => {
                    handleScan();
                  }}
                >
                  Scan
                </Button>
              </Card>
            </Grid>
          )}
          {loading && (
            <Box
              sx={{
                display: "flex",
                justifyContent: "center",
                alignItems: "center",
                minHeight: "60vh",
              }}
            >
              <CustomLoader />
            </Box>
          )}

          {step === 1 && (
            <Grid spacing={1} flex={"1 1 0"} flexDirection={"column"} container>
              <Button
                sx={{alignSelf: "start"}}
                startDecorator={<ArrowBackIcon />}
                onClick={() => {
                  setStep(0);
                  setImageResults(undefined);
                  setPrediction(undefined);
                }}
              >
                Check another
              </Button>
              <Card sx={{ padding: "1rem" }}>
                <Grid
                  container
                  sx={{ justifyContent: "space-between", alignItems: "center" }}
                >
                  {prediction && (
                    <Grid spacing={2} container>
                      {prediction === "Fake" ? (
                        <WarningOutlined />
                      ) : (
                        <CheckCircleIcon />
                      )}
                      {prediction && (
                        <Typography level="title-lg">
                          Prediction: {prediction}
                        </Typography>
                      )}
                    </Grid>
                  )}
                </Grid>
              </Card>
              <Card sx={{ padding: "1rem" }}>
                <Grid
                  spacing={1}
                  container
                  flex={"1 1 0"}
                  flexDirection={"column"}
                >
                  <Typography level="h4">Details</Typography>
                  {files.map((file, index) => (
                    <Grid size={6}>
                      <Box key={index}>
                        {file.type.startsWith("image/") && (
                          <img
                            src={URL.createObjectURL(file)}
                            alt={file.name}
                            style={{ width: "100%", height: "auto" }}
                          />
                        )}
                      </Box>
                    </Grid>
                  ))}
                </Grid>
                <Divider />
                <Grid spacing={2} flex={1} flexDirection={"row"} container>
                  {imageResults && (
                    <Grid size={{ xs: 12, md: 4 }}>
                      <Typography level="title-lg" marginBottom={1}>
                        Model Results
                      </Typography>
                      <Grid container>
                        <Grid size={6}>Deepfake</Grid>
                        <Grid size={6}>
                          {(imageResults.probability_fake * 100).toFixed(3)}%
                        </Grid>
                      </Grid>
                      <Grid container>
                        <Grid size={6}>Real</Grid>
                        <Grid size={6}>
                          {((1 - imageResults.probability_fake) * 100).toFixed(
                            3
                          )}
                          %
                        </Grid>
                      </Grid>
                    </Grid>
                  )}
                  <Grid size={{ xs: 12, md: 8 }}>
                    {imageResults && (
                      <PieChart
                        series={[
                          {
                            data: [
                              {
                                id: 1,
                                value: imageResults.probability_fake * 100,
                                label: "Deepfake",
                              },
                              {
                                id: 2,
                                value:
                                  (1 - imageResults.probability_fake) * 100,
                                label: "Real",
                              },
                            ],
                          },
                        ]}
                        width={400}
                        height={200}
                      />
                    )}
                  </Grid>
                  <Typography level="body-xs" sx={{ color: "text.secondary" }}>
                    *This detector's results should not be considered absolute
                    truth or definitive evidence.
                  </Typography>
                </Grid>
              </Card>
            </Grid>
          )}
        </Box>
      </Layout>
    </>
  );
};
