<?php

namespace Rubix\ML\Graph\Nodes;

use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Other\Helpers\Stats;
use Rubix\ML\Kernels\Distance\Distance;

use function Rubix\ML\argmax;

/**
 * Vantage Point
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class VantagePoint extends Ball
{
    /**
     * Factory method to build a hypersphere by splitting the dataset into
     * left and right clusters.
     *
     * @param \Rubix\ML\Datasets\Labeled $dataset
     * @param \Rubix\ML\Kernels\Distance\Distance $kernel
     * @return self
     */
    public static function split(Labeled $dataset, Distance $kernel) : self
    {
        $center = [];

        foreach ($dataset->columns() as $column => $values) {
            if ($dataset->columnType($column)->isContinuous()) {
                $center[] = Stats::mean($values);
            } else {
                $center[] = argmax(array_count_values($values));
            }
        }

        $distances = [];

        foreach ($dataset->samples() as $sample) {
            $distances[] = $kernel->compute($sample, $center);
        }

        $threshold = Stats::median($distances);

        $samples = $dataset->samples();
        $labels = $dataset->labels();

        $leftSamples = $leftLabels = $rightSamples = $rightLabels = [];

        foreach ($distances as $i => $distance) {
            if ($distance <= $threshold) {
                $leftSamples[] = $samples[$i];
                $leftLabels[] = $labels[$i];
            } else {
                $rightSamples[] = $samples[$i];
                $rightLabels[] = $labels[$i];
            }
        }

        $radius = max($distances) ?: 0.0;

        return new self($center, $radius, [
            Labeled::quick($leftSamples, $leftLabels),
            Labeled::quick($rightSamples, $rightLabels),
        ]);
    }
}
