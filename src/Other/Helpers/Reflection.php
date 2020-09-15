<?php

namespace Rubix\ML\Other\Helpers;

use ReflectionClass;
use ReflectionException;

/**
 * Reflection
 *
 * Utility functions relying on PHP's runtime reflection API
 *
 * @see https://www.php.net/manual/en/book.reflection.php
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Chris Simpson
 */
class Reflection
{
    /**
     * Property.
     *
     * Access a class property even if it's visibility is not public.
     * This is typically useful for debugging and creating informative logs.
     *
     * @param object $subject
     * @param string $name
     * @param mixed $fallback
     * @return mixed
     */
    public static function property(object $subject, string $name, $fallback = null)
    {
        try {
            $property = (new ReflectionClass($subject))->getProperty($name);
            $property->setAccessible(true);

            return $property->getValue($subject);
        } catch (ReflectionException $e) {
            return $fallback;
        }
    }
}
