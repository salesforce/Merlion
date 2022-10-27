/* resize figures in table upon callback get fires */

if(!window.dash_clientside) {window.dash_clientside = {};}
window.dash_clientside.clientside = {
   resize: function (value) {
       console.log("resizing...");
       window.dispatchEvent(new Event('resize'));
       return null
   }
}