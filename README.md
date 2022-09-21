# Quantum Community Detection for Recommender Systems

This repository contains the source code for the article
<a href="https://dl.acm.org/doi/abs/10.1145/3523227.3551478" target="_blank">
"Towards Recommender Systems with Community Detection and Quantum Computing"</a>
published at RecSys 2022. See the websites of our [quantum computing group](https://quantum.polimi.it/) for more
information on our teams and works.

Here we explain how to install dependencies, setup the connection to D-Wave Leap quantum cloud services and how to run
experiments included in this repository.

If you want to cite us or use our repository you can use the following bibtex entry:
```bibtex
@inproceedings{10.1145/3523227.3551478,
  author     = {Nembrini, Riccardo and Carugno, Costantino and {Ferrari Dacrema}, Maurizio and Cremonesi, Paolo},
  title      = {Towards Recommender Systems with Community Detection and Quantum Computing},
  year       = {2022},
  isbn       = {9781450392785},
  publisher  = {Association for Computing Machinery},
  address    = {New York, NY, USA},
  url        = {https://doi.org/10.1145/3523227.3551478},
  doi        = {10.1145/3523227.3551478},
  abstract   = {After decades of being mainly confined to theoretical research, Quantum Computing is now becoming a useful tool for solving realistic problems. This work aims to experimentally explore the feasibility of using currently available quantum computers, based on the Quantum Annealing paradigm, to build a recommender system exploiting community detection. Community detection, by partitioning users and items into densely connected clusters, can boost the accuracy of non-personalized recommendation by assuming that users within each community share similar tastes. However, community detection is a computationally expensive process. The recent availability of Quantum Annealers as cloud-based devices, constitutes a new and promising direction to explore community detection, although effectively leveraging this new technology is a long-term path that still requires advancements in both hardware and algorithms. This work aims to begin this path by assessing the quality of community detection formulated as a Quadratic Unconstrained Binary Optimization problem on a real recommendation scenario. Results on several datasets show that the quantum solver is able to detect communities of comparable quality with respect to classical solvers, but with better speedup, and the non-personalized recommendation models built on top of these communities exhibit improved recommendation quality. The takeaway is that quantum computing, although in its early stages of maturity and applicability, shows promise in its ability to support new recommendation models and to bring improved scalability as technology evolves.},
  booktitle  = {Proceedings of the 16th ACM Conference on Recommender Systems},
  pages      = {579–585},
  numpages   = {7},
  keywords   = {Quantum Computing, Quantum Annealing, Recommender Systems, Community Detection},
  location   = {Seattle, WA, USA},
  series     = {RecSys '22}
}
```

> DISCLAIMER: this is a work-in-progress repository, it may be updated soon with a newer version with better data and
> results saving.

## Installation

> NOTE: This repository requires Python 3.8

It is suggested to install all the required packages into a new Python environment. So, after repository checkout, enter
the repository folder and run the following commands to create a new environment:

If you're using `virtualenv`:

```bash
virtualenv -p python3 QACDRec
source QACDRec/bin/activate
```

If you're using `conda`:

```bash
conda create -n QACDRec python=3.8 anaconda
conda activate QACDRec
```

>Remember to add this project in the PYTHONPATH environmental variable if you plan to run the experiments 
on the terminal:
>```bash
>export PYTHONPATH=$PYTHONPATH:/path/to/project/folder
>```

Then, make sure you correctly activated the environment and install all the required packages through `pip`:

```bash
pip install -r requirements.txt
```

### Additional
If you want to experiment with personalized recommenders, not used in the published results, it is suggested to compile
Cython code in the repository.

In order to compile you must first have installed: `gcc` and `python3 dev`. Under Linux those can be installed with the
following commands:

```bash
sudo apt install gcc 
sudo apt-get install python3-dev
```

If you are using Windows as operating system, the installation procedure is a bit more complex. You may refer
to [THIS](https://github.com/cython/cython/wiki/InstallingOnWindows) guide.

Now you can compile all Cython algorithms by running the following command in the `./recsys` directory.
The script will compile within the current active environment. The code has been developed for Linux and Windows
platforms. During the compilation you may see some warnings.

```bash
python run_compile_all_cython.py
```

## D-Wave Setup

In order to make use of D-Wave cloud services you must first sign up to [D-Wave Leap](https://cloud.dwavesys.com/leap/)
and get your API token.

Then, you need to run the following command in the newly created Python environment:

```bash
dwave setup
```

This is a guided setup for D-Wave Ocean SDK. When asked to select non-open-source packages to install you should
answer `y` and install at least _D-Wave Drivers_ (the D-Wave Problem Inspector package is not required, but could be
useful to analyse problem solutions, if solving problems with the QPU only).

Then, continue the configuration by setting custom properties (or keeping the default ones, as we suggest), apart from
the `Authentication token` field, where you should paste your API token obtained on the D-Wave Leap dashboard.

You should now be able to connect to D-Wave cloud services. In order to verify the connection, you can use the following
command, which will send a test problem to D-Wave's QPU:

```bash
dwave ping
```

## Executing Experiments

In order to execute the experiments you must first run the community detection scripts:
- `run_community_detection.py` to perform community detection with D-Wave Leap Hybrid, Simulated Annealing, Steepest Descent and Tabu Search
- `qa_run_community_detection.py` to perform community detection with the D-Wave Advantage QPU, starting from iterations computed with the previous script

Then, you can run the recommendation experiments, which will build and evaluate TopPopular recommendation on the
communities found with the previous scripts:
- `cd_recommendation.py` to perform recommendation with the communities found with `run_community_detection.py`
- `qa_cd_recommendation.py` to perform recommendation with the communities found with `qa_run_community_detection.py`

> DISCLAIMER: performing the entire set of experiments can be expensive both in terms of classical computation time and
> in terms of D-Wave Leap time. Since the service offers 1 minute free per month, it is suggested to start running the
> experiments with a restricted number of datasets and samplers.
