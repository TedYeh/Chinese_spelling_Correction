var popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'))
var popoverList = popoverTriggerList.map(function (popoverTriggerEl) {
  return new bootstrap.Popover(popoverTriggerEl)
})
var popover = new bootstrap.Popover(document.querySelector('.output-container'), {
  container: 'body',
  html: true,
})

$('input[type=radio][name="models"]').on('change', function() {
  switch($(this).val()) {
    case 'Spelling-T5-Large':
      $("#prompt").hide()
      break
    case 'Spelling-T5-Base':
      $("#prompt").hide()
      break
    case 'Grammar-T5-Base':
      $("#prompt").show()
      break
  }      
})

$(document).ready(function() {
  $("#prompt").hide()
});

$('#submit').on('click', function() {
  var $this = $(this);
  //$this.attr('disabled', 'disabled');
  $('#submit').prepend("<span class='spinner-grow spinner-grow-sm' role='status' aria-hidden='true'></span>");
  $this.button('loading');
    setTimeout(function() {
      $this.button('reset');
  }, 8000);
});

function span_target(tar, loc_list){
  console.log(tar, loc_list);
  out_res = '<div>'
  for (let i=0; i < tar.length; i++) {
      if(loc_list.includes(i)){
          out_res = out_res.concat("<span class='btn btn-primary btn-sm'>"+tar[i]+"</span>")
      }else{
            out_res = out_res.concat(tar[i])
      }
  }
  out_res = out_res.concat('</div>')
  return out_res
}

/*
$(document).ready(function() {
  $('#out_result').html(span_target('吃了早餐以後他去上課。', [3]))
});*/


$("a").click(function(){
  i=$(this).data("value");
  $('#inputText').val(i);
});
/*
var popover = new bootstrap.Popover(document.querySelector('.popover-dismiss'), {
    trigger: 'focus'
})*/
