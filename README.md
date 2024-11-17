# Quick Start

## OpenAI Support
We use an OpenAI model to provide the ratings. This requires an OpenAI account
and API key. Export the key to the `OPENAI_API_KEY` environment variable:

```
export OPENAI_API_KEY='YOUR_KEY_GOES_HERE'
```

## Installing
This project uses Hatch to build and test the package. You'll need to install it. There are system-standard ways, but pip should work fine:

```
pip install hatch
```

Now you can test:

```
hatch test
```

To test the OpenAI integration, run with the "--external-model" option. These
tests cost a (small) amount of money and require an internet connection so we
don't run them by default.

```
hatch test --external-model
```

Now you can build and install it to your local Python installation (requires Python >= 3.10)

WARNING: If I'm too lazy to update the docs, just install whatever `*.whl` file is in the dist folder.

```
hatch build
pip install dist/ratemyhooptie-0.0.1-py3-none-any.whl
```

## Usage
You're now ready to run!

```
ratemyhooptie
```
