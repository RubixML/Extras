<?php

namespace Rubix\ML\Persisters {

    use Rubix\ML\Exceptions\RuntimeException;

    if (interface_exists('League\Flysystem\FilesystemOperator')) {
        class Flysystem extends \Rubix\ML\Persisters\FlysystemV2
        {
        }
    } elseif (interface_exists('League\Flysystem\FilesystemInterface')) {
        class Flysystem extends \Rubix\ML\Persisters\FlysystemV1
        {
        }
    } else {
        throw new RuntimeException('Flysystem dependency is not available.');
    }

}
