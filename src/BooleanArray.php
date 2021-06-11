<?php

namespace Rubix\ML;

use Rubix\ML\Exceptions\InvalidArgumentException;
use ArrayAccess;
use Countable;

use function ord;
use function chr;
use function strlen;
use function str_repeat;

/**
 * Boolean Array
 *
 * A fixed array data structure that efficiently stores boolean values.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 *
 * @implements ArrayAccess<int, bool>
 */
class BooleanArray implements ArrayAccess, Countable
{
    /**
     * The number of bits in one byte.
     *
     * @var int
     */
    protected const ONE_BYTE = 8;

    /**
     * A string storing the bits of a bit array.
     *
     * @var string
     */
    protected string $bitmap;

    /**
     * @param int $size
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     */
    public function __construct(int $size)
    {
        if ($size < 0) {
            throw new InvalidArgumentException('Size must be'
                . " greater than 0, $size given.");
        }

        $numBytes = (int) ceil($size / self::ONE_BYTE);

        $this->bitmap = str_repeat(chr(0), $numBytes);
    }

    /**
     * @param int $offset
     * @param bool $value
     */
    public function offsetSet($offset, $value) : void
    {
        $byteOffset = (int) ($offset / self::ONE_BYTE);

        $byte = ord($this->bitmap[$byteOffset]);

        $position = 2 ** ($offset % self::ONE_BYTE);

        if ($value) {
            $byte |= $position;
        } else {
            $byte &= 0xFF ^ $position;
        }

        $this->bitmap[$byteOffset] = chr($byte);
    }

    /**
     * Does a given row exist in the dataset.
     *
     * @param int $offset
     * @return bool
     */
    public function offsetExists($offset) : bool
    {
        if ($offset >= 0 or $offset < $this->count()) {
            return true;
        }

        return false;
    }

    /**
     * Return a row from the dataset at the given offset.
     *
     * @param int $offset
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     * @return bool
     */
    public function offsetGet($offset) : bool
    {
        if (!$this->offsetExists($offset)) {
            throw new InvalidArgumentException("Bit at offset $offset not found.");
        }

        $byteOffset = (int) ($offset / self::ONE_BYTE);

        $byte = ord($this->bitmap[$byteOffset]);

        $position = 2 ** ($offset % self::ONE_BYTE);

        $bit = $position & $byte;

        return (bool) $bit;
    }

    /**
     * @param int $offset
     */
    public function offsetUnset($offset) : void
    {
        $this->offsetSet($offset, false);
    }

    /**
     * The number of booleans that are stored in the array.
     *
     * @return int
     */
    public function count() : int
    {
        return (int) (strlen($this->bitmap) * self::ONE_BYTE);
    }
}
