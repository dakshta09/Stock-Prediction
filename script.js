function displayTable() {
            // Get the search term from the input field
	    var searchTerm = document.getElementById('searchInput').value;
            // Create a sample data array (you can replace this with your actual data)
            var data = [
                 {name: '', analysis: '',  invest: '', news:'', sentiment:'', projection:'' }  ];

            // Filter data based on the search term (you can customize this based on your needs)
            var filteredData = data.filter(function(item) {
                return item.name.toLowerCase().includes(searchTerm.toLowerCase());
            });

            // Create and populate the table
            var table = '<table>';
	    filteredData.forEach(function(item) {
            table += '<tr><th>Name:</th> <td>'+item.name+'</td></tr>';
	    table += '<tr><th>Analysis:</th> <td>'+item.analysis+'</td></tr>';
	    table += '<tr><th>Invest:</th> <td>'+item.invest+'</td></tr>';
	    table += '<tr><th>News:</th> <td>'+item.news+'</td></tr>';
	    table += '<tr><th>Sentiment Analysis:</th> <td>'+item.sentiment+'</td></tr>';
	    table += '<tr><th>One Year Projection:</th> <td>'+item.projection+'</td></tr>';

  table += '</tr>';
            });

            
            table += '</table>';

            // Display the table in the tableContainer div
            document.getElementById('tableContainer').innerHTML = table;
        }