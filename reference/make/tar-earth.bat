@set EARTH_VERSION=3.2-6

cd ..\..\standalone-earth\make
                        @if errorlevel 1 goto error
call make-clean
                        @if errorlevel 1 goto error
cd ..\..
                        @if errorlevel 1 goto error
@rem Use gnu tar because my ancient mks tar complains about filename too long
@rem tar -cf standalone-earth-%EARTH_VERSION%.tar standalone-earth
c:\Rtools\bin\tar -cf standalone-earth-%EARTH_VERSION%.tar standalone-earth
                        @if errorlevel 1 goto error
rm -f standalone-earth-%EARTH_VERSION%.tar.gz
                        @if errorlevel 1 goto error
gzip standalone-earth-%EARTH_VERSION%.tar
                        @if errorlevel 1 goto error
mv standalone-earth-%EARTH_VERSION%.tar.gz /a/homepage/earth
                        @if errorlevel 1 goto error
ls -l /a/homepage/earth/standalone*
                        @if errorlevel 1 goto error

cd standalone-earth\make
@if errorlevel 1 goto error

@goto done
:error
@echo ==== ERROR ====
:done
