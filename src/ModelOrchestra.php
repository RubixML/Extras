<?php

namespace Rubix\ML;

use Rubix\ML\Backends\Serial;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Regressors\Ridge;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Other\Helpers\Params;
use Rubix\ML\Backends\Tasks\Proba;
use Rubix\ML\Backends\Tasks\Predict;
use Rubix\ML\Other\Traits\LoggerAware;
use Rubix\ML\Other\Traits\PredictsSingle;
use Rubix\ML\Backends\Tasks\TrainLearner;
use Rubix\ML\Other\Traits\Multiprocessing;
use Rubix\ML\Classifiers\SoftmaxClassifier;
use Rubix\ML\Other\Traits\AutotrackRevisions;
use Rubix\ML\Specifications\DatasetIsNotEmpty;
use Rubix\ML\Specifications\SpecificationChain;
use Rubix\ML\Specifications\SamplesAreCompatibleWithEstimator;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;

use function count;
use function in_array;

/**
 * Model Orchestra
 *
 * Model Orchestra is a stacking ensemble that uses a separate model, called the
 * *conductor*, to learn the influences of each member of the ensemble.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class ModelOrchestra implements Learner, Parallel, Persistable, Verbose
{
    use AutotrackRevisions, Multiprocessing, PredictsSingle, LoggerAware;

    /**
     * The members of the orchestra.
     *
     * @var \Rubix\ML\Learner[]
     */
    protected $members;

    /**
     * The learner responsible for making the final prediction given the predictions
     * of the members.
     *
     * @var \Rubix\ML\Learner
     */
    protected $conductor;

    /**
     * The ratio of training samples used to train the members.
     *
     * @var float
     */
    protected $ratio;

    /**
     * The data types that the orchestra is compatible with.
     *
     * @var \Rubix\ML\DataType[]
     */
    protected $compatibility;

    /**
     * @param \Rubix\ML\Learner[] $members
     * @param \Rubix\ML\Learner|null $conductor
     * @param float $ratio
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     */
    public function __construct(array $members, ?Learner $conductor = null, float $ratio = 0.8)
    {
        if (empty($members)) {
            throw new InvalidArgumentException('Orchestra must contain at'
                . ' least 1 member.');
        }

        $proto = current($members);

        $compatibilities = [];

        foreach ($members as $estimator) {
            if (!$estimator instanceof Learner) {
                throw new InvalidArgumentException('Member must implement'
                    . ' the Learner interface.');
            }

            $type = $estimator->type();

            if (!$type->isClassifier() and !$type->isRegressor()) {
                throw new InvalidArgumentException('Orchestra is only'
                    . ' compatible with classifiers and regressors, '
                    . " $type given.");
            }

            if ($type->isClassifier() and !$estimator instanceof Probabilistic) {
                throw new InvalidArgumentException('Classifier must implement'
                    . ' the Probabilistic interface.');
            }

            if ($type != $proto->type()) {
                throw new InvalidArgumentException('Members must all be'
                    . " of the same estimator type, {$proto->type()}"
                    . " expected but $type given.");
            }

            $compatibilities[] = $estimator->compatibility();
        }

        $compatibility = array_values(array_intersect(...$compatibilities));

        if (count($compatibility) < 1) {
            throw new InvalidArgumentException('Members must have at'
                . ' least 1 compatible data type in common.');
        }

        if ($conductor) {
            if ($conductor->type() != $proto->type()) {
                throw new InvalidArgumentException('Conductor must be the'
                    . " same type as the members, {$proto->type()} expected"
                    . " but {$conductor->type()} given.");
            }

            if (!in_array(DataType::continuous(), $conductor->compatibility())) {
                throw new InvalidArgumentException('Conductor must be'
                    . ' compatible with continuous data types.');
            }
        } else {
            switch ($proto->type()) {
                case EstimatorType::classifier():
                    $conductor = new SoftmaxClassifier();

                    break;

                case EstimatorType::regressor():
                    $conductor = new Ridge();

                    break;

                default:
                    $conductor = new Ridge();
            }
        }

        if ($ratio <= 0.0 or $ratio >= 1.0) {
            throw new InvalidArgumentException('Ratio must be strictly'
                . " between 0 and 1, $ratio given.");
        }

        $this->members = $members;
        $this->conductor = $conductor;
        $this->ratio = $ratio;
        $this->compatibility = $compatibility;
        $this->backend = new Serial();
    }

    /**
     * Return the integer encoded estimator type.
     *
     * @return \Rubix\ML\EstimatorType
     */
    public function type() : EstimatorType
    {
        return $this->conductor->type();
    }

    /**
     * Return the data types that this estimator is compatible with.
     *
     * @return \Rubix\ML\DataType[]
     */
    public function compatibility() : array
    {
        return $this->compatibility;
    }

    /**
     * Has the learner been trained?
     *
     * @return bool
     */
    public function trained() : bool
    {
        foreach ($this->members as $member) {
            if (!$member->trained()) {
                return false;
            }
        }

        return $this->conductor->trained();
    }

    /**
     * Return the settings of the hyper-parameters in an associative array.
     *
     * @return mixed[]
     */
    public function params() : array
    {
        return [
            'members' => $this->members,
            'conductor' => $this->conductor,
            'ratio' => $this->ratio,
        ];
    }

    /**
     * Return the estimators that comprise the member part of the ensemble.
     *
     * @return \Rubix\ML\Learner[]
     */
    public function members() : array
    {
        return $this->members;
    }

    /**
     * Return the conductor of the ensemble.
     *
     * @return \Rubix\ML\Learner
     */
    public function conductor() : Learner
    {
        return $this->conductor;
    }

    /**
     * Instantiate and train each base estimator in the ensemble on a bootstrap
     * training set.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     */
    public function train(Dataset $dataset) : void
    {
        if (!$dataset instanceof Labeled) {
            throw new InvalidArgumentException('Learner requires a'
                . ' Labeled training set.');
        }

        SpecificationChain::with([
            new DatasetIsNotEmpty($dataset),
            new SamplesAreCompatibleWithEstimator($dataset, $this),
        ])->check();

        if ($this->logger) {
            $this->logger->info("$this initialized");
        }

        [$left, $right] = $dataset->labelType()->isCategorical()
            ? $dataset->stratifiedSplit($this->ratio)
            : $dataset->randomize()->split($this->ratio);

        $this->backend->flush();

        foreach ($this->members as $estimator) {
            $this->backend->enqueue(
                new TrainLearner($estimator, $left),
                [$this, 'afterTrain']
            );
        }

        $this->members = $this->backend->process();

        $right = Labeled::quick($this->extract($right), $right->labels());

        if ($this->logger) {
            $this->logger->info('Learning member influences');
        }

        $this->conductor->train($right);

        if ($this->logger) {
            $this->logger->info('Training complete');
        }
    }

    /**
     * Make predictions from a dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @throws \Rubix\ML\Exceptions\RuntimeException
     * @return mixed[]
     */
    public function predict(Dataset $dataset) : array
    {
        if (!$this->trained()) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        $dataset = Unlabeled::quick($this->extract($dataset));

        return $this->conductor->predict($dataset);
    }

    /**
     * The callback that executes after the training task.
     *
     * @param \Rubix\ML\Learner $estimator
     * @throws \Rubix\ML\Exceptions\RuntimeException
     */
    public function afterTrain(Learner $estimator) : void
    {
        if (!$estimator->trained()) {
            throw new RuntimeException('There was a problem training '
                . Params::shortName(get_class($estimator)) . '.');
        }

        if ($this->logger) {
            $this->logger->info(Params::shortName(get_class($estimator))
                . ' finished training');
        }
    }

    /**
     * Extract features from the ensemble members.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @return array[]
     */
    protected function extract(Dataset $dataset) : array
    {
        switch ($this->type()) {
            case EstimatorType::classifier():
                return $this->extractClassifier($dataset);

            case EstimatorType::regressor():
            default:
                return $this->extractRegressor($dataset);
        }
    }

    /**
     * Extract features from an ensemble of classifiers.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @return array[]
     */
    protected function extractClassifier(Dataset $dataset) : array
    {
        $this->backend->flush();

        foreach ($this->members as $estimator) {
            if ($estimator instanceof Probabilistic) {
                $this->backend->enqueue(new Proba($estimator, $dataset));
            }
        }

        $aggregate = array_transpose($this->backend->process());

        $samples = [];

        foreach ($aggregate as $probabilities) {
            $samples[] = array_merge(...array_map('array_values', $probabilities));
        }

        return $samples;
    }

    /**
     * Extract features from an ensemble of regressors.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @return array[]
     */
    protected function extractRegressor(Dataset $dataset) : array
    {
        $this->backend->flush();

        foreach ($this->members as $estimator) {
            $this->backend->enqueue(new Predict($estimator, $dataset));
        }

        return array_transpose($this->backend->process());
    }

    /**
     * Return the string representation of the object.
     *
     * @return string
     */
    public function __toString() : string
    {
        return 'Model Orchestra (' . Params::stringify($this->params()) . ')';
    }
}
