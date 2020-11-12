echo "Make sure to install python 3.8+ before proceeding with this script"
echo "Enter full path for the python environment: "
read location
echo "Entered path is `$location`"
python3 -m venv $location
env_name=`basename $location`
echo "Activating environment"
source $env_name/bin/activate
python_location=`which python | grep $location/bin/python`
location=$location/bin/python
if [[ $python_location == $location ]]
then
	echo "Proceeding further. Environment successfully created"
else
	echo "Entered wrong path of location. Try again and delete the environment folder if created at wrong location"
	exit 0
	fi
echo "Upgrading PIP"
pip install --upgrade pip
echo "Installing modules and its dependencies"
pip install -r requirements.txt
echo "Deactivating environment"
deactivate
echo "Done"
exit 0


