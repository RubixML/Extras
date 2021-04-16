<span style="float:right;"><a href="https://github.com/RubixML/Extras/blob/master/src/Serializers/Igbinary.php">[source]</a></span>

# Igbinary
Igbinary is a compact binary format that serves as a drop-in replacement for the native PHP serializer.

!!! note
    The [Igbinary extension](https://github.com/igbinary/igbinary) is needed to use this serializer.

## Parameters
This serializer does not have any parameters.

## Example
```php
use Rubix\ML\Persisters\Serializers\Igbinary;

$serializer = new Igbinary();
```