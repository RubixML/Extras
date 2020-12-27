<?php

namespace Rubix\ML\Kernels\Distance;

use Rubix\ML\DataType;

/**
 * Levenshtein
 *
 * Levenshtein distance is defined as the number of single-character edits (such as insert, delete,
 * or replace) needed to change one word to another.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Levenshtein implements Distance
{
    /**
     * Return the data types that this kernel is compatible with.
     *
     * @return \Rubix\ML\DataType[]
     */
    public function compatibility() : array
    {
        return [
            DataType::categorical(),
        ];
    }

    /**
     * Compute the distance between two vectors.
     *
     * @param list<string> $a
     * @param list<string> $b
     * @return float
     */
    public function compute(array $a, array $b) : float
    {
        return (float) array_sum(array_map('levenshtein', $a, $b));
    }

    /**
     * Return the string representation of the object.
     *
     * @return string
     */
    public function __toString() : string
    {
        return 'Levenshtein';
    }
}
