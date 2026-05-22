import express from "express";
import cors from "cors";
import path from "path";
import fs from "fs";
import { exec } from "child_process";
import { fileURLToPath } from "url";
import { PNG } from "pngjs";

const app = express();
const PORT = Number(process.env.PORT || 5000);

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PROJECT_ROOT = path.resolve(__dirname, "..");
const GENERATED_DIR = path.join(__dirname, "generated-results");
const MPIEXEC = process.env.MPIEXEC_PATH || "mpiexec";

fs.mkdirSync(GENERATED_DIR, { recursive: true });

app.use(cors());
app.use(express.json());
app.use("/results", express.static(GENERATED_DIR));
app.use(express.static(PROJECT_ROOT));

function quoteCommandArg(value) {
  if (/^".*"$/.test(value)) {
    return value;
  }

  if (/^[A-Za-z0-9_./:-]+$/.test(value)) {
    return value;
  }

  return `"${value.replace(/"/g, '\\"')}"`;
}

function execPromise(command, cwd) {
  return new Promise((resolve, reject) => {
    exec(
      command,
      {
        cwd,
        windowsHide: true,
        maxBuffer: 20 * 1024 * 1024,
      },
      (error, stdout, stderr) => {
        if (error) {
          error.stdout = stdout;
          error.stderr = stderr;
          reject(error);
          return;
        }

        resolve({ stdout, stderr });
      },
    );
  });
}

function tokenizePgmHeader(buffer) {
  let index = 0;
  const tokens = [];

  while (index < buffer.length && tokens.length < 4) {
    while (
      index < buffer.length &&
      /\s/.test(String.fromCharCode(buffer[index]))
    ) {
      index += 1;
    }

    if (buffer[index] === 35) {
      while (index < buffer.length && buffer[index] !== 10) {
        index += 1;
      }
      continue;
    }

    const start = index;
    while (
      index < buffer.length &&
      !/\s/.test(String.fromCharCode(buffer[index]))
    ) {
      index += 1;
    }

    tokens.push(buffer.toString("ascii", start, index));
  }

  while (
    index < buffer.length &&
    /\s/.test(String.fromCharCode(buffer[index]))
  ) {
    index += 1;
  }

  return { tokens, dataOffset: index };
}

function pgmToPng(pgmPath, pngPath) {
  const buffer = fs.readFileSync(pgmPath);
  const { tokens, dataOffset } = tokenizePgmHeader(buffer);

  if (tokens.length < 4) {
    throw new Error(`Invalid PGM header in ${pgmPath}`);
  }

  const [magic, widthToken, heightToken, maxvalToken] = tokens;
  if (magic !== "P5") {
    throw new Error(`Unsupported PGM format in ${pgmPath}: ${magic}`);
  }

  const width = Number(widthToken);
  const height = Number(heightToken);
  const maxval = Number(maxvalToken);

  if (
    !Number.isFinite(width) ||
    !Number.isFinite(height) ||
    !Number.isFinite(maxval)
  ) {
    throw new Error(`Malformed PGM dimensions in ${pgmPath}`);
  }

  if (maxval <= 0 || maxval > 255) {
    throw new Error(`Only 8-bit P5 PGM files are supported: ${pgmPath}`);
  }

  const expectedPixels = width * height;
  const pixels = buffer.subarray(dataOffset, dataOffset + expectedPixels);

  if (pixels.length < expectedPixels) {
    throw new Error(`PGM pixel data is truncated in ${pgmPath}`);
  }

  const png = new PNG({ width, height });

  for (let i = 0; i < expectedPixels; i += 1) {
    const value = pixels[i];
    const pixelIndex = i * 4;
    png.data[pixelIndex] = value;
    png.data[pixelIndex + 1] = value;
    png.data[pixelIndex + 2] = value;
    png.data[pixelIndex + 3] = 255;
  }

  fs.writeFileSync(pngPath, PNG.sync.write(png));
}

function parseExecutionTime(stdout) {
  const patterns = [
    /Execution time\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)\s*(ms|milliseconds?|s|sec|secs|seconds?)?/i,
    /Serial time\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)\s*(ms|milliseconds?|s|sec|secs|seconds?)?/i,
    /OpenMP time\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)\s*(ms|milliseconds?|s|sec|secs|seconds?)?/i,
    /MPI time\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)\s*(ms|milliseconds?|s|sec|secs|seconds?)?/i,
    /GPU total time\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)\s*ms(?:\s*\(([0-9]+(?:\.[0-9]+)?)\s*sec\))?/i,
    /GPU kernel exec\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)\s*(ms|milliseconds?|s|sec|secs|seconds?)?/i,
  ];

  for (const pattern of patterns) {
    const match = stdout.match(pattern);
    if (!match) {
      continue;
    }

    const value = Number(match[1]);
    const unit = (match[2] || "").toLowerCase();
    const alternateSeconds = match[3] ? Number(match[3]) : null;

    if (Number.isFinite(alternateSeconds)) {
      return alternateSeconds;
    }

    if (unit.startsWith("ms")) {
      return value / 1000;
    }

    return value;
  }

  return null;
}

const METHODS = {
  serial: {
    label: "Serial",
    cwd: path.join(PROJECT_ROOT, "Serial"),
    command: "serial_conv.exe input.pgm output_serial.pgm",
    outputPgm: path.join(PROJECT_ROOT, "Serial", "output_serial.pgm"),
    outputPng: path.join(GENERATED_DIR, "serial.png"),
  },
  openmp: {
    label: "OpenMP",
    cwd: path.join(PROJECT_ROOT, "OpenMP"),
    command: "openmp_conv.exe input.pgm output_openmp.pgm",
    outputPgm: path.join(PROJECT_ROOT, "OpenMP", "output_openmp.pgm"),
    outputPng: path.join(GENERATED_DIR, "openmp.png"),
  },
  mpi: {
    label: "MPI",
    cwd: path.join(PROJECT_ROOT, "MPI"),
    command: `${quoteCommandArg(MPIEXEC)} -n 4 mpi_conv.exe input.pgm output_mpi.pgm`,
    outputPgm: path.join(PROJECT_ROOT, "MPI", "output_mpi.pgm"),
    outputPng: path.join(GENERATED_DIR, "mpi.png"),
  },
  cuda: {
    label: "CUDA",
    cwd: path.join(PROJECT_ROOT, "CUDA"),
    command: "cuda_conv.exe input.pgm output_custom.pgm",
    outputPgm: path.join(PROJECT_ROOT, "CUDA", "output_custom.pgm"),
    outputPng: path.join(GENERATED_DIR, "cuda.png"),
  },
};

async function runMethod(methodKey, res) {
  const method = METHODS[methodKey];

  if (!method) {
    res.status(404).json({ success: false, error: "Unknown method" });
    return;
  }

  try {
    const { stdout, stderr } = await execPromise(method.command, method.cwd);
    const executionTimeSeconds = parseExecutionTime(stdout);

    if (executionTimeSeconds === null) {
      res.status(500).json({
        success: false,
        error: `Could not find an execution time in ${method.label} stdout`,
        stdout,
        stderr,
      });
      return;
    }

    pgmToPng(method.outputPgm, method.outputPng);

    res.json({
      success: true,
      method: methodKey,
      message: `${method.label} completed successfully.`,
      executionTimeSeconds,
      executionTimeLabel: `${executionTimeSeconds.toFixed(6)} seconds`,
      imageUrl: `/results/${methodKey}.png`,
      stdout,
      stderr,
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      error: error.message || `Failed to run ${method.label}`,
      stdout: error.stdout || "",
      stderr: error.stderr || "",
    });
  }
}

app.get("/api/health", (_req, res) => {
  res.json({ success: true, message: "HPC demo backend is running." });
});

app.post("/api/run/serial", (_req, res) => {
  void runMethod("serial", res);
});

app.post("/api/run/openmp", (_req, res) => {
  void runMethod("openmp", res);
});

app.post("/api/run/mpi", (_req, res) => {
  void runMethod("mpi", res);
});

app.post("/api/run/cuda", (_req, res) => {
  void runMethod("cuda", res);
});

app.listen(PORT, () => {
  console.log(`HPC demo backend running at http://localhost:${PORT}`);
  console.log(`Serving project assets from ${PROJECT_ROOT}`);
});
